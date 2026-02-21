.PHONY: help build seedvr2-docker-build seedvr2-docker-run seedvr2-doctor docker-gpu-doctor setup-docker-gpu setup-docker-gpu-wsl

ROOT_DIR := $(CURDIR)
SEEDVR2_IMAGE := seedvr2-upscaler
SEEDVR2_INPUT_DIR := $(ROOT_DIR)/sandbox/inputs
SEEDVR2_OUTPUT_DIR := $(ROOT_DIR)/sandbox/outputs
SEEDVR2_CKPTS_DIR := $(ROOT_DIR)/.seedvr2-ckpts
SEEDVR2_ARGS ?=
SEEDVR2_VAE_SPLIT_SIZE ?= 1
SEEDVR2_VAE_CONV_MAX_MEM ?= 0.05
SEEDVR2_VAE_NORM_MAX_MEM ?= 0.05
SEEDVR2_VAE_MEMORY_DEVICE ?= cpu
SEEDVR2_MAX_PIXELS ?= 0
SEEDVR2_PYTORCH_CUDA_ALLOC_CONF ?= expandable_segments:True
SEEDVR2_TILE_MODE ?= off
SEEDVR2_TILE_OVERLAP ?= 64
SEEDVR2_TILE_GRID ?= 3

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  build                 Clone SeedVR2 dependency into dependencies/seedvr2' \
	  '  seedvr2-docker-build   Build the SeedVR2 Docker image' \
	  '  seedvr2-docker-run     Run the SeedVR2 Docker image (GPU required)' \
	  '  seedvr2-doctor         Check local GPU + Docker prerequisites' \
	  '  setup-docker-gpu       Attempt to install/configure NVIDIA Container Toolkit' \
	  '  setup-docker-gpu-wsl   End-to-end GPU setup for Docker on WSL/rootless/rootful'

build:
	@if [ -d dependencies/seedvr2 ]; then \
		printf '%s\n' 'dependencies/seedvr2 already exists; skipping clone'; \
	else \
		git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler dependencies/seedvr2 && \
		rm -rf dependencies/seedvr2/.git; \
	fi

seedvr2-docker-build:
	docker build -f docker/seedvr2/Dockerfile -t $(SEEDVR2_IMAGE) .

seedvr2-doctor:
	@set -eu; \
	printf '%s\n' '== GPU (host) =='; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi >/dev/null 2>&1 && printf '%s\n' 'nvidia-smi: OK' || (printf '%s\n' 'nvidia-smi: FAILED'; exit 1); \
	else \
		printf '%s\n' 'nvidia-smi: missing (install NVIDIA driver)'; \
		exit 1; \
	fi; \
	printf '%s\n' '' '== Docker =='; \
	command -v docker >/dev/null 2>&1 || (printf '%s\n' 'docker: missing'; exit 1); \
	docker version >/dev/null 2>&1 && printf '%s\n' 'docker: OK' || (printf '%s\n' 'docker: FAILED'; exit 1); \
	docker_ctx=$$(docker context show 2>/dev/null || true); \
	[ -n "$$docker_ctx" ] && printf '%s\n' "docker context: $$docker_ctx" || true; \
	printf '%s\n' '' '== NVIDIA runtime (Docker) =='; \
	if docker info 2>/dev/null | grep -iE 'Runtimes:.*nvidia|nvidia-container-runtime' >/dev/null 2>&1; then \
		printf '%s\n' 'nvidia runtime: detected in docker info'; \
	else \
		printf '%s\n' 'nvidia runtime: NOT detected'; \
		if [ "$$docker_ctx" = 'rootless' ]; then \
			printf '%s\n' \
			  'Fix: run `make setup-docker-gpu`, then restart rootless Docker (often: `systemctl --user restart docker`).'; \
		else \
			printf '%s\n' \
			  'Fix: run `make setup-docker-gpu`, then restart Docker (often: `sudo systemctl restart docker`).'; \
		fi; \
		exit 2; \
	fi

setup-docker-gpu:
	@set -eu; \
	if ! command -v docker >/dev/null 2>&1; then \
		printf '%s\n' 'docker not found; install Docker first.'; \
		exit 1; \
	fi; \
	docker_ctx=$$(docker context show 2>/dev/null || true); \
	if docker info 2>/dev/null | grep -iE 'Runtimes:.*nvidia|nvidia-container-runtime' >/dev/null 2>&1; then \
		printf '%s\n' 'NVIDIA runtime already present according to `docker info`.'; \
		exit 0; \
	fi; \
	if command -v nvidia-ctk >/dev/null 2>&1; then \
		printf '%s\n' 'Configuring Docker runtime via nvidia-ctk...'; \
		if [ "$$docker_ctx" = 'rootless' ]; then \
			cfg="$${XDG_CONFIG_HOME:-$$HOME/.config}/docker/daemon.json"; \
			mkdir -p "$$(dirname "$$cfg")"; \
			printf '%s\n' "Using rootless daemon config: $$cfg"; \
			nvidia-ctk runtime configure --runtime=docker --config="$$cfg"; \
			if command -v systemctl >/dev/null 2>&1; then \
				systemctl --user restart docker 2>/dev/null || systemctl --user restart docker.service 2>/dev/null || true; \
			fi; \
			printf '%s\n' 'Done. Restart rootless Docker if needed (often: `systemctl --user restart docker`).'; \
		else \
			sudo nvidia-ctk runtime configure --runtime=docker; \
			if command -v systemctl >/dev/null 2>&1; then \
				sudo systemctl restart docker; \
			fi; \
			printf '%s\n' 'Done. Restart Docker if needed (often: `sudo systemctl restart docker`).'; \
		fi; \
		printf '%s\n' 'Re-run `make seedvr2-doctor`.'; \
		exit 0; \
	fi; \
	if command -v apt-get >/dev/null 2>&1; then \
		printf '%s\n' 'Installing NVIDIA Container Toolkit (apt-based distro)...'; \
		distribution=$$(. /etc/os-release && printf '%s' "$$ID$$VERSION_ID"); \
		curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - >/dev/null; \
		curl -fsSL "https://nvidia.github.io/nvidia-docker/$$distribution/nvidia-docker.list" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list >/dev/null; \
		sudo apt-get update; \
		sudo apt-get install -y nvidia-container-toolkit; \
		if command -v nvidia-ctk >/dev/null 2>&1; then \
			sudo nvidia-ctk runtime configure --runtime=docker; \
		fi; \
		sudo systemctl restart docker; \
		printf '%s\n' 'Done. Re-run `make seedvr2-doctor`.'; \
		exit 0; \
	fi; \
	printf '%s\n' \
	  'Unsupported package manager for automatic install.' \
	  'Install NVIDIA Container Toolkit manually:' \
	  '  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html' \
	  'Then run:' \
	  '  sudo nvidia-ctk runtime configure --runtime=docker' \
	  '  sudo systemctl restart docker'

setup-docker-gpu-wsl:
	@set -eu; \
	if ! command -v nvidia-smi >/dev/null 2>&1; then \
		printf '%s\n' 'nvidia-smi not found in WSL. Ensure NVIDIA driver + WSL GPU support are installed on Windows.'; \
		exit 1; \
	fi; \
	nvidia-smi >/dev/null 2>&1 || (printf '%s\n' 'nvidia-smi failed; fix host GPU driver setup first.'; exit 1); \
	if ! command -v docker >/dev/null 2>&1; then \
		printf '%s\n' 'docker not found; install Docker first.'; \
		exit 1; \
	fi; \
	if ! command -v nvidia-ctk >/dev/null 2>&1; then \
		if command -v apt-get >/dev/null 2>&1; then \
			printf '%s\n' 'Installing NVIDIA Container Toolkit (apt-based distro)...'; \
			distribution=$$(. /etc/os-release && printf '%s' "$$ID$$VERSION_ID"); \
			curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - >/dev/null; \
			curl -fsSL "https://nvidia.github.io/nvidia-docker/$$distribution/nvidia-docker.list" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list >/dev/null; \
			sudo apt-get update; \
			sudo apt-get install -y nvidia-container-toolkit; \
		else \
			printf '%s\n' 'nvidia-ctk missing and apt-get unavailable. Install toolkit manually:'; \
			printf '%s\n' 'https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html'; \
			exit 1; \
		fi; \
	fi; \
	docker_ctx=$$(docker context show 2>/dev/null || true); \
	if [ "$$docker_ctx" = 'rootless' ]; then \
		cfg="$${XDG_CONFIG_HOME:-$$HOME/.config}/docker/daemon.json"; \
		mkdir -p "$$(dirname "$$cfg")"; \
		printf '%s\n' "Configuring rootless Docker runtime in $$cfg"; \
		nvidia-ctk runtime configure --runtime=docker --config="$$cfg"; \
		if [ -f /etc/nvidia-container-runtime/config.toml ]; then \
			printf '%s\n' 'Enabling no-cgroups=true for rootless NVIDIA runtime compatibility...'; \
			sudo sed -i 's/^#\?no-cgroups = .*/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml; \
		fi; \
		if command -v systemctl >/dev/null 2>&1; then \
			systemctl --user restart docker 2>/dev/null || systemctl --user restart docker.service 2>/dev/null || true; \
		fi; \
	else \
		printf '%s\n' 'Configuring rootful Docker runtime...'; \
		sudo nvidia-ctk runtime configure --runtime=docker; \
		if command -v systemctl >/dev/null 2>&1; then \
			sudo systemctl restart docker; \
		fi; \
	fi; \
	printf '%s\n' 'Running final checks...'; \
	$(MAKE) seedvr2-doctor; \
	docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi >/dev/null && \
	printf '%s\n' 'GPU container smoke test: OK' || \
	(printf '%s\n' 'GPU container smoke test: FAILED'; exit 1)

seedvr2-docker-run:
	@mkdir -p "$(SEEDVR2_INPUT_DIR)" "$(SEEDVR2_OUTPUT_DIR)" "$(SEEDVR2_CKPTS_DIR)";
	docker run --rm --gpus all \
	  -e PYTORCH_CUDA_ALLOC_CONF="$(SEEDVR2_PYTORCH_CUDA_ALLOC_CONF)" \
	  -e SEEDVR2_VAE_SPLIT_SIZE="$(SEEDVR2_VAE_SPLIT_SIZE)" \
	  -e SEEDVR2_VAE_CONV_MAX_MEM="$(SEEDVR2_VAE_CONV_MAX_MEM)" \
	  -e SEEDVR2_VAE_NORM_MAX_MEM="$(SEEDVR2_VAE_NORM_MAX_MEM)" \
	  -e SEEDVR2_VAE_MEMORY_DEVICE="$(SEEDVR2_VAE_MEMORY_DEVICE)" \
	  -e SEEDVR2_MAX_PIXELS="$(SEEDVR2_MAX_PIXELS)" \
	  -e SEEDVR2_TILE_MODE="$(SEEDVR2_TILE_MODE)" \
	  -e SEEDVR2_TILE_OVERLAP="$(SEEDVR2_TILE_OVERLAP)" \
	  -e SEEDVR2_TILE_GRID="$(SEEDVR2_TILE_GRID)" \
	  --mount type=bind,source="$(SEEDVR2_INPUT_DIR)",target=/input,readonly \
	  --mount type=bind,source="$(SEEDVR2_OUTPUT_DIR)",target=/output \
	  --mount type=bind,source="$(SEEDVR2_CKPTS_DIR)",target=/opt/seedvr/ckpts \
	  $(SEEDVR2_IMAGE) --input /input --output /output --scale 2 $(SEEDVR2_ARGS)
