# Convenience wrapper around the CMake/TT-Metal build.
#
# Requires: $TT_METAL_HOME exported (run `source $TT_METAL_HOME/setup.sh`
# first). The CMakeLists in host/ picks up TT-Metal's exported targets.
#
# Targets:
#   make all     — configure + build the host driver
#   make run     — alias: build then python bench_blackhole.py with sane defaults
#   make profile — same, with TT-Metal device profiler enabled
#   make clean   — remove the build/ tree (also gitignored)
#   make -n      — dry-run; useful from a non-TT host as a smoke check

BUILD_DIR := build
HOST_BIN  := $(BUILD_DIR)/bench_blackhole

OUT       ?= ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_$(shell date +%Y%m%d).jsonl
LAYERS    ?= A,B,C
SIZES     ?= 512,1024,2048,4096,8192

.PHONY: all run profile clean

all: $(HOST_BIN)

$(HOST_BIN): host/main.cpp host/CMakeLists.txt kernels/*.cpp
	@mkdir -p $(BUILD_DIR)
	@if [ -z "$$TT_METAL_HOME" ]; then \
		echo "TT_METAL_HOME is not set. Run 'source \$$TT_METAL_HOME/setup.sh' first." >&2; \
		exit 1; \
	fi
	cmake -S host -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

run: all
	python bench_blackhole.py --out $(OUT) --layers $(LAYERS) --sizes $(SIZES)

profile: all
	TT_METAL_DEVICE_PROFILER=1 python bench_blackhole.py \
		--out $(OUT) --layers $(LAYERS) --sizes $(SIZES) --profile

clean:
	rm -rf $(BUILD_DIR)
