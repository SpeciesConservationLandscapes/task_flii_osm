FROM debian:bullseye-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends osmium-tool python3 python3-pip git python3-gdal gdal-bin curl apt-transport-https ca-certificates gnupg wget && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install \
    requests \
    tqdm \
    earthengine-api \
    google-cloud-storage \
    google-api-python-client

WORKDIR /app
COPY src ./src