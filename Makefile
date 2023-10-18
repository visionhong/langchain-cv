NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
        # NVCC not found
        USE_CUDA := 0
        NVCC_VERSION := "not installed"
else
        NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
        USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
        TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
        BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
        TORCH_CUDA_ARCH_LIST :=
        BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif


build-image:
        @echo $(BUILD_MESSAGE)
        docker build --build-arg USE_CUDA=$(USE_CUDA) \
        --build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
        --no-cache -t langchain-cv/torch:2.0.1 .

run:
        docker run -d --gpus all -it --net=host --privileged \
        -e DISPLAY=$DISPLAY \
        -p 8507:8507 \
        -v /home/jeff/study/checkpoints:/volume/checkpoints \
        --name=langchain-cv \
        --ipc=host -it langchain-cv/torch:2.0.1

remove:
        docker stop langchain-cv && docker rm langchain-cv