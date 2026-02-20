FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone IDM-VTON HuggingFace Space first (has preprocessing code + ckpts)
RUN git lfs install && \
    git clone https://huggingface.co/spaces/yisol/IDM-VTON /app/idm-vton

WORKDIR /app/idm-vton

# Install ALL deps from the space's own requirements.txt (skip torch* and pillow - already in base)
RUN sed -e '/^torch/d' -e '/^pillow/d' requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

# RunPod SDK + brotli fix + detectron2 from source
RUN pip install --no-cache-dir runpod brotlicffi && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Download IDM-VTON model weights from HuggingFace model repo
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('yisol/IDM-VTON', local_dir='/app/models/IDM-VTON')"

# Copy custom handler (replaces Gradio app)
COPY handler.py /app/idm-vton/handler.py

CMD ["python", "handler.py"]
