FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch: the default wheel drags in several GB of CUDA libraries.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the detector weights into the image so the container starts offline.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY app ./app

EXPOSE 8095
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8095"]
