RIFE_IMAGE ?= video-pipelines-rife:latest
ESRGAN_IMAGE ?= video-pipelines-esrgan:latest
SEEDVR2_IMAGE ?= video-pipelines-seedvr2:latest
DEPTH_ANYTHING_V2_IMAGE ?= video-pipelines-depth-anything-v2:latest
DEPTH_ANYTHING_V3_IMAGE ?= video-pipelines-depth-anything-v3:latest
DEPTH_ANYTHING_V3_STREAMING_IMAGE ?= video-pipelines-depth-anything-v3-streaming:latest
TORCH_CHANNELS ?= cu128 cu129 cu124 cu121 nightly/cu128 nightly/cu129
DEPTH_CRAFTER_IMAGE ?= video-pipelines-depth-crafter:latest
DEPTH_PRO_IMAGE ?= video-pipelines-depth-pro:latest
NORMAL_CRAFTER_IMAGE ?= video-pipelines-normal-crafter:latest
DKT_NORMAL_IMAGE ?= video-pipelines-dkt-normal:latest
DOCKER_GPU_ARGS ?= --gpus all
RIFE_MODEL_CACHE_DIR ?= .cache/rife-model
ESRGAN_MODEL_CACHE_DIR ?= .cache/esrgan-model
SEEDVR2_MODEL_CACHE_DIR ?= .cache/seedvr2-model
DEPTH_CRAFTER_MODEL_CACHE_DIR ?= .cache/depth-crafter
DEPTH_ANYTHING_V3_STREAMING_MODEL_CACHE_DIR ?= .cache/depth-anything-v3-streaming
DEPTH_PRO_MODEL_CACHE_DIR ?= .cache/depth-pro
NORMAL_CRAFTER_MODEL_CACHE_DIR ?= .cache/normal-crafter
DKT_NORMAL_MODEL_CACHE_DIR ?= .cache/dkt-normal-model
ARTIFACT_DIR ?= data
PIPELINE_DB_PATH ?= .video_pipelines.duckdb

BUILD_DIR ?= $(CURDIR)/.build
CUDSS_VERSION ?= 0.5.0.16
CERES_REF ?= master
COLMAP_VERSION ?= 4.0.3
VENV_PYTHON ?= $(CURDIR)/.venv/bin/python
CUDA_ARCH_LIST ?= all-major


.PHONY: clean-all cli \
	install-cudss build-ceres build-colmap build-pycolmap build-all

clean-all:
	rm -rf "$(ARTIFACT_DIR)"
	rm -f "$(PIPELINE_DB_PATH)" "$(PIPELINE_DB_PATH).wal"

# ── COLMAP with cuDSS (GPU bundle adjustment) ───────────────────────────────
# Installs cuDSS, rebuilds Ceres + COLMAP + pycolmap into $(BUILD_DIR).
# Run once: make build-all
# Then restart your shell so the new pycolmap is picked up.

build-all: install-cudss build-ceres build-colmap build-pycolmap
	@echo "Done. cuDSS-enabled pycolmap is installed into .venv"

install-cudss:
	@echo "==> Installing cuDSS $(CUDSS_VERSION)"
	@mkdir -p "$(BUILD_DIR)/cudss"
	@URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-$(CUDSS_VERSION)_cuda12-archive.tar.xz"; \
	curl -fL "$$URL" -o "$(BUILD_DIR)/cudss/cudss.tar.xz"; \
	tar -xf "$(BUILD_DIR)/cudss/cudss.tar.xz" -C "$(BUILD_DIR)/cudss" --strip-components=1
	@sudo cp -r "$(BUILD_DIR)/cudss/lib/"* /usr/local/lib/
	@sudo cp -r "$(BUILD_DIR)/cudss/include/"* /usr/local/include/
	@sudo ldconfig

build-ceres:
	@echo "==> Building Ceres ($(CERES_REF)) with cuDSS"
	@sudo apt-get install -y --no-install-recommends \
		cmake ninja-build git libgoogle-glog-dev libgflags-dev \
		libeigen3-dev libsuitesparse-dev
	@rm -rf "$(BUILD_DIR)/ceres"
	@git clone --depth 1 --branch "$(CERES_REF)" --recurse-submodules \
		https://github.com/ceres-solver/ceres-solver.git "$(BUILD_DIR)/ceres"
	@cmake -S "$(BUILD_DIR)/ceres" -B "$(BUILD_DIR)/ceres-build" -GNinja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DCMAKE_PREFIX_PATH="/usr/local;/usr/local/lib/cmake;/usr/local/lib64/cmake" \
		-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH_LIST) \
		-DBUILD_TESTING=OFF \
		-DBUILD_EXAMPLES=OFF \
		-DUSE_CUDA=static \
		-DCUDA_ARCHS=$(CUDA_ARCH_LIST)
	@ninja -C "$(BUILD_DIR)/ceres-build"
	@grep -q "CERES_NO_CUDSS" "$(BUILD_DIR)/ceres-build/include/ceres/internal/config.h" && \
		(echo "Ceres was built without cuDSS support. Check cuDSS installation and CMake detection." && exit 1) || true
	@sudo ninja -C "$(BUILD_DIR)/ceres-build" install
	@sudo ldconfig

build-colmap:
	@echo "==> Building COLMAP $(COLMAP_VERSION)"
	@sudo apt-get install -y --no-install-recommends \
		libboost-all-dev libflann-dev libfreeimage-dev libmetis-dev \
		libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
		libopenimageio-dev openimageio-tools libopencv-dev
	@mkdir -p "$(BUILD_DIR)/colmap"
	@curl -fL "https://github.com/colmap/colmap/archive/refs/tags/$(COLMAP_VERSION).tar.gz" \
		| tar -xz -C "$(BUILD_DIR)/colmap" --strip-components=1
	@sed -i 's|https://github.com/PoseLib/PoseLib/archive/f119951fca625133112acde48daffa5f20eba451.zip|https://codeload.github.com/PoseLib/PoseLib/zip/f119951fca625133112acde48daffa5f20eba451|g' "$(BUILD_DIR)/colmap/src/thirdparty/CMakeLists.txt"
	@cmake -S "$(BUILD_DIR)/colmap" -B "$(BUILD_DIR)/colmap-build" -GNinja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH_LIST) \
		-DCUDA_ENABLED=ON \
		-DCUDA_ARCHS=$(CUDA_ARCH_LIST) \
		-DGUI_ENABLED=OFF
	@ninja -C "$(BUILD_DIR)/colmap-build"
	@sudo ninja -C "$(BUILD_DIR)/colmap-build" install
	@sudo ldconfig

build-pycolmap: build-colmap
	@echo "==> Building pycolmap against local COLMAP source"
	@test -f "$(BUILD_DIR)/colmap/pyproject.toml" || (echo "Missing $(BUILD_DIR)/colmap/pyproject.toml. Run: make build-colmap" && exit 1)
	@"$(VENV_PYTHON)" -m pip uninstall -y pycolmap pycolmap-cuda12 2>/dev/null || true
	@CUDACXX=/usr/local/cuda/bin/nvcc \
		CUDA_HOME=/usr/local/cuda \
		PATH=/usr/local/cuda/bin:$$PATH \
		"$(VENV_PYTHON)" -m pip install "$(BUILD_DIR)/colmap" \
		--config-settings=cmake.args="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc;-DCUDA_ENABLED=ON;-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH_LIST);-DCUDA_ARCHS=$(CUDA_ARCH_LIST)"

CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

cli:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "Virtualenv not found at $(VENV_PYTHON). Running uv sync once..."; \
		uv sync; \
	fi
	"$(VENV_PYTHON)" src/pipe/cli.py $(CLI_ARGS)

%:
	@:
