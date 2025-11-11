FROM debian:bookworm-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        osmium-tool \
        python3 python3-pip \
        git \
        python3-gdal gdal-bin libgdal-dev \
        curl apt-transport-https ca-certificates gnupg wget coreutils \
        procps psmisc lsof && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install \
    requests==2.32.5 \
    tqdm==4.67.1 \
    earthengine-api==1.6.11 \
    google-cloud-storage==3.4.1 \
    google-api-python-client==2.184.0 \
    packaging==25.0  \
    importlib-metadata>=6.6  # harmless on 3.11, but fine to keep

WORKDIR /app
COPY src ./src

