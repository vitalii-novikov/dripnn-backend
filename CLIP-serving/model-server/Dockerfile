FROM pytorch/torchserve:latest-cpu

RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.41.0 \
    Pillow \
    numpy \
    torchvision

COPY model-store /home/model-server/model-store
COPY config.properties /home/model-server/

EXPOSE 8080 8081

CMD ["torchserve", "--start",\
     "--model-store", "/home/model-server/model-store", \
     "--ts-config", "/home/model-server/config.properties"]
