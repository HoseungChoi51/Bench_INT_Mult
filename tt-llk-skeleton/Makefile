# Convenience wrapper around the CMake/TT-Metal build.
#
# Requires: $TT_METAL_HOME exported. If unset, falls back to ~/TT/tt-metal
# which matches the layout on the dev box. Override on the command line:
#   make TT_METAL_HOME=/opt/tt-metal all
#
# Targets:
#   make all     — configure + build the host driver
#   make run     — alias: build then python bench_blackhole.py with sane defaults
#   make profile — same, with TT-Metal device profiler enabled
#   make clean   — remove the build/ tree (gitignored)
#   make -n      — dry-run; useful from a non-TT host

TT_METAL_HOME ?= $(HOME)/TT/tt-metal
ARCH_NAME     ?= blackhole
export TT_METAL_HOME
export ARCH_NAME

BUILD_DIR := build
HOST_BIN  := $(BUILD_DIR)/bench_blackhole

OUT       ?= ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_$(shell date +%Y%m%d).jsonl
LAYERS    ?= A,B,C
SIZES     ?= 512,1024,2048,4096,8192

.PHONY: all run profile clean check-env

check-env:
	@if [ ! -d "$(TT_METAL_HOME)" ]; then \
		echo "TT_METAL_HOME=$(TT_METAL_HOME) does not exist." >&2; exit 1; \
	fi

all: $(HOST_BIN)

$(HOST_BIN): host/main.cpp host/CMakeLists.txt $(wildcard kernels/*.cpp) | check-env
	@mkdir -p $(BUILD_DIR)
	cmake -S host -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

run: all
	python bench_blackhole.py --out $(OUT) --layers $(LAYERS) --sizes $(SIZES)

profile: all
	TT_METAL_DEVICE_PROFILER=1 python bench_blackhole.py \
		--out $(OUT) --layers $(LAYERS) --sizes $(SIZES) --profile

clean:
	rm -rf $(BUILD_DIR)
