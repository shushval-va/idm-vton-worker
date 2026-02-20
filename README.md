# IDM-VTON RunPod Serverless Worker

Virtual try-on endpoint using [IDM-VTON](https://github.com/yisol/IDM-VTON) (ECCV 2024).

Features:
- **Auto-masking**: DensePose + human parsing + OpenPose (no manual mask needed)
- **SDXL quality**: 768x1024 output resolution
- **Auto-crop**: Handles any aspect ratio input

## Deploy on RunPod

1. Create a GitHub repo with this Dockerfile + handler.py
2. In RunPod console: Serverless → New Endpoint → GitHub Integration
3. Select repo, set GPU to 24GB+ (RTX 4090 or A5000)
4. Deploy

## API Usage

```python
import requests, base64

API_KEY = "your_runpod_api_key"
ENDPOINT_ID = "your_endpoint_id"

with open("person.jpg", "rb") as f:
    person_b64 = base64.b64encode(f.read()).decode()
with open("garment.jpg", "rb") as f:
    garment_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "person_image": person_b64,
            "garment_image": garment_b64,
            "garment_description": "black lace lingerie set",
            "category": "upper_body",
        }
    },
    timeout=300,
)
result = resp.json()["output"]["image"]
with open("result.png", "wb") as f:
    f.write(base64.b64decode(result))
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| person_image | string | required | Base64 encoded person photo |
| garment_image | string | required | Base64 encoded garment photo |
| garment_description | string | "clothing" | Text description of garment |
| category | string | "upper_body" | "upper_body", "lower_body", or "dresses" |
| denoise_steps | int | 30 | Denoising steps (20-40) |
| seed | int | 42 | Random seed |
| auto_crop | bool | true | Auto-crop to 3:4 aspect ratio |
