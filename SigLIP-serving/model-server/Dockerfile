FROM pytorch/torchserve:latest

WORKDIR /home/model-server

COPY model-store/open_clip_pytorch_model.bin model-store/
COPY model-store/config.json model-store/
COPY model-store/preprocessor_config.json model-store/
COPY siglip_handler.py model-store/

COPY config.properties .

CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "siglip=model-store", "--ts-config", "config.properties"]
