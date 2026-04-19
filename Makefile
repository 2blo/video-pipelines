RIFE_IMAGE ?= video-pipelines-rife:latest
ESRGAN_IMAGE ?= video-pipelines-esrgan:latest
DOCKER_GPU_ARGS ?= --gpus all
RIFE_MODEL_CACHE_DIR ?= .cache/rife-model
ESRGAN_MODEL_CACHE_DIR ?= .cache/esrgan-model
ARTIFACT_DIR ?= data
PIPELINE_DB_PATH ?= .video_pipelines.duckdb

BUILD_DIR ?= $(CURDIR)/.build
CUDSS_VERSION ?= 0.5.0.16
CERES_REF ?= master
COLMAP_VERSION ?= 4.0.3
VENV_PYTHON ?= $(CURDIR)/.venv/bin/python
CUDA_ARCH_LIST ?= all-major

.PHONY: rife-image rife-upscale rife-example esrgan-image esrgan-upscale clean-all \
	install-cudss build-ceres build-colmap build-pycolmap build-colmap-cuda build-all

rife-image:
	docker build -t $(RIFE_IMAGE) -f docker/rife/Dockerfile .

rife-upscale:
	@if [ -z "$(INPUT)" ] || [ -z "$(SCALE)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make rife-upscale INPUT=/path/in.mp4 SCALE=2 OUTPUT=/path/out.mp4"; \
		exit 1; \
	fi
	@if ! docker image inspect "$(RIFE_IMAGE)" >/dev/null 2>&1; then \
		$(MAKE) rife-image; \
	fi
	@in_abs="$$(realpath -m "$(INPUT)")"; \
	out_abs="$$(realpath -m "$(OUTPUT)")"; \
	in_dir="$$(dirname "$$in_abs")"; \
	out_dir="$$(dirname "$$out_abs")"; \
	cache_dir="$$(realpath -m "$(RIFE_MODEL_CACHE_DIR)")"; \
	mkdir -p "$$cache_dir"; \
	mkdir -p "$$out_dir"; \
	docker run --rm $(DOCKER_GPU_ARGS) \
		-v "$$in_dir:/io/in:ro" \
		-v "$$out_dir:/io/out" \
		-v "$$cache_dir:/opt/rife/train_log" \
		$(RIFE_IMAGE) \
		"/io/in/$$(basename "$$in_abs")" \
		"$(SCALE)" \
		"/io/out/$$(basename "$$out_abs")"

rife-example: rife-image
	@mkdir -p sandbox/inputs sandbox/outputs
	@docker run --rm --entrypoint ffmpeg \
		-v "$(CURDIR)/sandbox/inputs:/io" \
		$(RIFE_IMAGE) \
		-y -f lavfi -i testsrc=size=1280x720:rate=24 -t 3 -pix_fmt yuv420p /io/example.mp4
	@$(MAKE) rife-upscale \
		INPUT="$(CURDIR)/sandbox/inputs/example.mp4" \
		SCALE=2 \
		OUTPUT="$(CURDIR)/sandbox/outputs/example_2x.mp4"

esrgan-image:
	docker build -t $(ESRGAN_IMAGE) -f docker/esrgan/Dockerfile .

esrgan-upscale:
	@if [ -z "$(INPUT)" ] || [ -z "$(WIDTH)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make esrgan-upscale INPUT=/path/in.mp4 WIDTH=2160 OUTPUT=/path/out.mkv"; \
		exit 1; \
	fi
	@if ! docker image inspect "$(ESRGAN_IMAGE)" >/dev/null 2>&1; then \
		$(MAKE) esrgan-image; \
	fi
	@in_abs="$$(realpath -m "$(INPUT)")"; \
	out_abs="$$(realpath -m "$(OUTPUT)")"; \
	in_dir="$$(dirname "$$in_abs")"; \
	out_dir="$$(dirname "$$out_abs")"; \
	cache_dir="$$(realpath -m "$(ESRGAN_MODEL_CACHE_DIR)")"; \
	mkdir -p "$$cache_dir"; \
	mkdir -p "$$out_dir"; \
	docker run --rm $(DOCKER_GPU_ARGS) \
		-v "$$in_dir:/io/in:ro" \
		-v "$$out_dir:/io/out" \
		-v "$$cache_dir:/opt/esrgan/models" \
		$(ESRGAN_IMAGE) \
		"/io/in/$$(basename "$$in_abs")" \
		"$(WIDTH)" \
		"/io/out/$$(basename "$$out_abs")"

clean-all:
	rm -rf "$(ARTIFACT_DIR)"
	rm -f "$(PIPELINE_DB_PATH)" "$(PIPELINE_DB_PATH).wal"

# ── COLMAP with cuDSS (GPU bundle adjustment) ───────────────────────────────
# Installs cuDSS, rebuilds Ceres + COLMAP + pycolmap into $(BUILD_DIR).
# Run once: make build-all
# Then restart your shell so the new pycolmap is picked up.

build-all: install-cudss build-ceres build-colmap build-pycolmap
	@echo "Done. cuDSS-enabled pycolmap is installed into .venv"

build-colmap-cuda: build-all

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

make cli:
	uv run src/pipe/cli.py
