FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (matching IDM-VTON requirements)
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    scipy==1.10.1 \
    scikit-image==0.21.0 \
    opencv-python==4.7.0.72 \
    config==0.5.1 \
    einops==0.7.0 \
    onnxruntime==1.16.2 \
    basicsr \
    fvcore \
    cloudpickle \
    omegaconf \
    pycocotools \
    huggingface_hub==0.25.0

# Install detectron2 from source (needs CUDA devel headers)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Clone IDM-VTON HuggingFace Space (has preprocessing code + ckpts)
RUN git lfs install && \
    git clone https://huggingface.co/spaces/yisol/IDM-VTON /app/idm-vton

WORKDIR /app/idm-vton

# Download IDM-VTON model weights from HuggingFace model repo
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('yisol/IDM-VTON', local_dir='/app/models/IDM-VTON')"

# Copy custom handler (replaces Gradio app)
COPY handler.py /app/idm-vton/handler.py

CMD ["python", "handler.py"]
