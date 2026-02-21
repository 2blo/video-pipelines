build:
# 	clone to folder dependencies/seedvr2
	git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler dependencies/seedvr2
	rm -rf dependencies/seedvr2/.git

ifeq ($(wildcard /dev/dxg),/dev/dxg)
VIDEO2X_DOCKER_FLAGS ?= -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
else
VIDEO2X_DOCKER_FLAGS ?= --gpus all
endif

VIDEO2X_DOCKER_RUN = docker run --rm $(VIDEO2X_DOCKER_FLAGS) -v $$(pwd):/host video2x-local:6.4.0
VIDEO2X_GPU_DOCKER_FLAGS = --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all
VIDEO2X_GPU_DOCKER_RUN = docker run --rm $(VIDEO2X_GPU_DOCKER_FLAGS) -v $$(pwd):/host video2x-local:6.4.0

build-video2x-image:
	docker build -f Dockerfile.video2x -t video2x-local:6.4.0 .

video2x-4k-pass:
	$(VIDEO2X_DOCKER_RUN) -i sandbox/inputs/second_step_0_Trim.mp4 -o sandbox/outputs/second_step_0_Trim_4k.mp4 -p libplacebo -w 3840 -h 2160 --libplacebo-shader anime4k-v4-a+a

video2x-4k-pass-gpu:
	$(VIDEO2X_GPU_DOCKER_RUN) -i sandbox/inputs/second_step_0_Trim.mp4 -o sandbox/outputs/second_step_0_Trim_4k.mp4 -p libplacebo -w 3840 -h 2160 --libplacebo-shader anime4k-v4-a+a

video2x-120fps-pass:
	$(VIDEO2X_DOCKER_RUN) -i sandbox/outputs/second_step_0_Trim_4k.mp4 -o sandbox/outputs/second_step_0_Trim_4k_120fps.mp4 -p rife -m 4 --rife-model rife-v4.6

video2x-120fps-pass-gpu:
	$(VIDEO2X_GPU_DOCKER_RUN) -i sandbox/outputs/second_step_0_Trim_4k.mp4 -o sandbox/outputs/second_step_0_Trim_4k_120fps.mp4 -p rife -m 4 --rife-model rife-v4.6

video2x-list-devices:
	$(VIDEO2X_GPU_DOCKER_RUN) --list-devices
