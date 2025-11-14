FROM pytorch/torchserve:latest-cpu

WORKDIR /model-server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model-store/ model-store/
COPY fashionclip_handler.py .
COPY config.properties .

# TorchServe will automatically start via CMD in docker-compose
