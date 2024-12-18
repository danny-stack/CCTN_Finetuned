# Use PyTorch as base image with CUDA 11.6
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    FORCE_CUDA="1"

WORKDIR /workspace/CCTN_Finetune

# Install system dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    git wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    numpy==1.24.4 pillow==9.4.0 matplotlib==3.7.5 scipy==1.10.1 \
    pandas==2.0.3 pycocotools==2.0.7 opencv-python==4.10.0.84 \
    mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html \
    mmdet==2.28.1 mmengine==0.1.0 gdown addict==2.4.0 \
    terminaltables==3.1.10 PyYAML==6.0.2 tqdm==4.65.2 \
    rich==13.4.2 requests==2.28.2 fastapi uvicorn \
    torchaudio==0.13.1 torchvision==0.14.1

# Create directories
RUN mkdir -p /workspace/CCTN_Finetune/CCTN_dataset/{train,valid,test,annotations} \
    && mkdir -p /workspace/CCTN_Finetune/{checkpoints,configs,tools,imgs,CCTN_dataset,work_dirs}

# Copy project files, including mmdetection and CascadeTabNet
# COPY . /workspace/CCTN_Finetune/
COPY ./mmdetection /workspace/CCTN_Finetune/mmdetection/
COPY ./CascadeTabNet /workspace/CCTN_Finetune/CascadeTabNet/
# COPY ./tools /workspace/CCTN_Finetune/tools
# COPY ./checkpoints /workspace/CCTN_Finetune/checkpoints

# COPY ./mmdetection /workspace/CCTN_Finetune/mmdetection/
# COPY ./CascadeTabNet /workspace/CCTN_Finetune/CascadeTabNet/
# RUN mkdir -p /workspace/CCTN_Finetune/{configs,tools,checkpoints,CCTN_dataset,work_dirs}

# Set permissions
RUN chmod -R 755 /workspace/CCTN_Finetune

# Default command
# CMD ["python", "api.py"]

# Expose the port
EXPOSE 6006

# Start the FastAPI server
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "6006"]

COPY run.sh /workspace/CCTN_Finetune/
RUN chmod +x /workspace/CCTN_Finetune/run.sh

CMD ["./run.sh", "api"]  

# Build the image
# docker build -t cctn_finetune:latest .

# Run the training script (Windows)
# docker run --gpus all ^
# -v %cd%/checkpoints:/workspace/CCTN_Finetune/checkpoints ^
# -v %cd%/CCTN_dataset:/workspace/CCTN_Finetune/CCTN_dataset ^
# -v %cd%/work_dirs:/workspace/CCTN_Finetune/work_dirs ^
# cctn_finetune:latest ./run.sh train

# docker run --gpus all ^
# -v %cd%/configs:/workspace/CCTN_Finetune/configs ^
# -v %cd%/tools:/workspace/CCTN_Finetune/tools ^
# -v %cd%/checkpoints:/workspace/CCTN_Finetune/checkpoints ^
# -v %cd%/CCTN_dataset:/workspace/CCTN_Finetune/CCTN_dataset ^
# -v %cd%/work_dirs:/workspace/CCTN_Finetune/work_dirs ^
# -v %cd%/run.sh:/workspace/CCTN_Finetune/run.sh ^
# cctn_finetune:latest ./run.sh eval

# Linux
# docker run --gpus all \
# -v $(pwd)/checkpoints:/workspace/CCTN_Finetune/checkpoints \
# -v $(pwd)/CCTN_dataset:/workspace/CCTN_Finetune/CCTN_dataset \
# -v $(pwd)/work_dirs:/workspace/CCTN_Finetune/work_dirs \
# cctn_finetune:latest ./run.sh train