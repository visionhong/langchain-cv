FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.7/

RUN mkdir -p /home/appuser/langchain-cv

COPY . /home/appuser/langchain-cv

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* \
    vim=2:* curl -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get -qq install nodejs --yes

WORKDIR /home/appuser/langchain-cv/frontend
RUN npm i && npm run build

WORKDIR /home/appuser/langchain-cv
RUN pip install --no-cache-dir -r requirements.txt