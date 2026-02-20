"""RunPod serverless handler for IDM-VTON virtual try-on.

Accepts person_image + garment_image (base64), runs auto-masking
and IDM-VTON inference, returns result image (base64).
"""

import base64
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import runpod

# IDM-VTON code is at /app/idm-vton (cloned from HF Space)
sys.path.insert(0, "/app/idm-vton")

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
import apply_net

# ---------------------------------------------------------------------------
# Model loading (runs once at worker startup)
# ---------------------------------------------------------------------------
MODEL_PATH = "/app/models/IDM-VTON"

print("Loading IDM-VTON models...")

unet = UNet2DConditionModel.from_pretrained(
    MODEL_PATH, subfolder="unet", torch_dtype=torch.float16
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    MODEL_PATH, subfolder="tokenizer", use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
    MODEL_PATH, subfolder="tokenizer_2", use_fast=False
)

noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.float16
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    MODEL_PATH, subfolder="text_encoder_2", torch_dtype=torch.float16
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    MODEL_PATH, subfolder="image_encoder", torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    MODEL_PATH, subfolder="vae", torch_dtype=torch.float16
)

unet_encoder = UNet2DConditionModel_ref.from_pretrained(
    MODEL_PATH, subfolder="unet_encoder", torch_dtype=torch.float16
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

for m in [unet_encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
    m.requires_grad_(False)

tensor_tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

pipe = TryonPipeline.from_pretrained(
    MODEL_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = unet_encoder

print("IDM-VTON models loaded.")


# ---------------------------------------------------------------------------
# Inference logic (adapted from HF Space app.py)
# ---------------------------------------------------------------------------

def run_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str = "clothing",
    category: str = "upper_body",
    denoise_steps: int = 30,
    seed: int = 42,
    auto_crop: bool = True,
) -> Image.Image:
    device = "cuda"

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm = garment_img.convert("RGB").resize((768, 1024))
    human_orig = person_img.convert("RGB")

    # Optional auto-crop to 3:4 aspect ratio
    if auto_crop:
        w, h = human_orig.size
        tw = int(min(w, h * (3 / 4)))
        th = int(min(h, w * (4 / 3)))
        left = (w - tw) / 2
        top = (h - th) / 2
        right = (w + tw) / 2
        bottom = (h + th) / 2
        cropped = human_orig.crop((left, top, right, bottom))
        crop_size = cropped.size
        human_img = cropped.resize((768, 1024))
    else:
        human_img = human_orig.resize((768, 1024))

    # --- Auto-masking ---
    small = human_img.resize((384, 512))
    keypoints = openpose_model(small)
    model_parse, _ = parsing_model(small)
    mask, _ = get_mask_location("hd", category, model_parse, keypoints)
    mask = mask.resize((768, 1024))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_tf(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # --- DensePose ---
    human_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_arg = convert_PIL_to_numpy(human_arg, format="BGR")

    dp_args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    pose_img = dp_args.func(dp_args, human_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # --- IDM-VTON inference ---
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt = "model is wearing " + garment_desc
        neg = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg,
        )

        prompt_c = "a photo of " + garment_desc
        prompt_embeds_c, _, _, _ = pipe.encode_prompt(
            [prompt_c],
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=[neg],
        )

        pose_t = tensor_tf(pose_img).unsqueeze(0).to(device, torch.float16)
        garm_t = tensor_tf(garm).unsqueeze(0).to(device, torch.float16)

        generator = (
            torch.Generator(device).manual_seed(seed) if seed is not None else None
        )

        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=neg_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled.to(device, torch.float16),
            negative_pooled_prompt_embeds=neg_pooled.to(device, torch.float16),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_t,
            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
            cloth=garm_t,
            mask_image=mask,
            image=human_img,
            height=1024,
            width=768,
            ip_adapter_image=garm.resize((768, 1024)),
            guidance_scale=2.0,
        )[0]

    if auto_crop:
        out = images[0].resize(crop_size)
        human_orig.paste(out, (int(left), int(top)))
        return human_orig
    return images[0]


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(event):
    """
    Input JSON:
      person_image: base64 encoded person photo
      garment_image: base64 encoded garment photo
      garment_description: text description (default "clothing")
      category: "upper_body" | "lower_body" | "dresses" (default "upper_body")
      denoise_steps: int (default 30)
      seed: int (default 42)
      auto_crop: bool (default true)
    """
    try:
        inp = event["input"]

        person_img = Image.open(
            io.BytesIO(base64.b64decode(inp["person_image"]))
        )
        garment_img = Image.open(
            io.BytesIO(base64.b64decode(inp["garment_image"]))
        )

        result = run_tryon(
            person_img=person_img,
            garment_img=garment_img,
            garment_desc=inp.get("garment_description", "clothing"),
            category=inp.get("category", "upper_body"),
            denoise_steps=int(inp.get("denoise_steps", 30)),
            seed=int(inp.get("seed", 42)),
            auto_crop=inp.get("auto_crop", True),
        )

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
