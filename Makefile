RIFE_IMAGE ?= video-pipelines-rife:latest
DOCKER_GPU_ARGS ?= --gpus all
ARTIFACT_DIR ?= data
PIPELINE_DB_PATH ?= .video_pipelines.duckdb

.PHONY: rife-image rife-upscale rife-example clean-all

rife-image:
	docker build -t $(RIFE_IMAGE) -f docker/rife/Dockerfile .

rife-upscale: rife-image
	@if [ -z "$(INPUT)" ] || [ -z "$(SCALE)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make rife-upscale INPUT=/path/in.mp4 SCALE=2 OUTPUT=/path/out.mp4"; \
		exit 1; \
	fi
	@in_abs="$$(realpath -m "$(INPUT)")"; \
	out_abs="$$(realpath -m "$(OUTPUT)")"; \
	in_dir="$$(dirname "$$in_abs")"; \
	out_dir="$$(dirname "$$out_abs")"; \
	mkdir -p "$$out_dir"; \
	docker run --rm $(DOCKER_GPU_ARGS) \
		-v "$$in_dir:/io/in:ro" \
		-v "$$out_dir:/io/out" \
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

clean-all:
	rm -rf "$(ARTIFACT_DIR)"
	rm -f "$(PIPELINE_DB_PATH)" "$(PIPELINE_DB_PATH).wal"
