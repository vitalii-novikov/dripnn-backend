FROM pytorch/torchserve:latest

WORKDIR /home/model-server

COPY model-store/ model-store/
COPY siglip_handler.py .
COPY config.properties .

EXPOSE 8080 8081

CMD ["torchserve", "--start", "--ts-config", "config.properties"]