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

BUILD_DIR      := build
HOST_BIN       := $(BUILD_DIR)/bench_blackhole
SFPU_BUILD_DIR := build_sfpu
SFPU_BIN       := $(SFPU_BUILD_DIR)/bench_sfpu_int32_fma
IP_BUILD_DIR   := build_sfpu_ip
IP_BIN         := $(IP_BUILD_DIR)/bench_sfpu_int32_inner_product
INT8_BUILD_DIR := build_int8_tuned
INT8_BIN       := $(INT8_BUILD_DIR)/bench_int8_tuned

OUT       ?= ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_$(shell date +%Y%m%d).jsonl
LAYERS    ?= A,B,C
SIZES     ?= 512,1024,2048,4096,8192

.PHONY: all run profile clean check-env bench_sfpu_int32_fma sfpu_run \
        bench_sfpu_int32_inner_product ip_run \
        bench_int8_tuned int8_tuned_run

check-env:
	@if [ ! -d "$(TT_METAL_HOME)" ]; then \
		echo "TT_METAL_HOME=$(TT_METAL_HOME) does not exist." >&2; exit 1; \
	fi

all: $(HOST_BIN)

$(HOST_BIN): host/main.cpp host/CMakeLists.txt $(wildcard kernels/*.cpp) | check-env
	@mkdir -p $(BUILD_DIR)
	cmake -S host -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

# Phase 8 — SFPU INT32 fused mul+add benchmark.
bench_sfpu_int32_fma: $(SFPU_BIN)

$(SFPU_BIN): host_sfpu/main.cpp host_sfpu/CMakeLists.txt $(wildcard kernels/*.cpp) | check-env
	@mkdir -p $(SFPU_BUILD_DIR)
	cmake -S host_sfpu -B $(SFPU_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(SFPU_BUILD_DIR) --parallel

sfpu_run: bench_sfpu_int32_fma
	uv run --directory .. python scripts/bench_blackhole_sfpu_int32_fma.py \
		--out ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_sfpu_int32_fma_$(shell date +%Y%m%d).jsonl

# Phase 8 (extension) — INT32 inner-product benchmark on the SFPU.
bench_sfpu_int32_inner_product: $(IP_BIN)

$(IP_BIN): host_sfpu_ip/main.cpp host_sfpu_ip/CMakeLists.txt $(wildcard kernels/*.cpp) | check-env
	@mkdir -p $(IP_BUILD_DIR)
	cmake -S host_sfpu_ip -B $(IP_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(IP_BUILD_DIR) --parallel

ip_run: bench_sfpu_int32_inner_product
	uv run --directory .. python scripts/bench_blackhole_sfpu_int32_ip.py \
		--out ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_sfpu_int32_ip_$(shell date +%Y%m%d).jsonl

# Phase 7 (extension) — INT8 tuned matmul (block-tiled, no mcast).
bench_int8_tuned: $(INT8_BIN)

$(INT8_BIN): host_int8_tuned/main.cpp host_int8_tuned/CMakeLists.txt | check-env
	@mkdir -p $(INT8_BUILD_DIR)
	cmake -S host_int8_tuned -B $(INT8_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(INT8_BUILD_DIR) --parallel

int8_tuned_run: bench_int8_tuned
	uv run --directory .. python scripts/bench_blackhole_int8_tuned.py \
		--out ../bench-results/blackhole_$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)_int8_tuned_$(shell date +%Y%m%d).jsonl

run: all
	python bench_blackhole.py --out $(OUT) --layers $(LAYERS) --sizes $(SIZES)

profile: all
	TT_METAL_DEVICE_PROFILER=1 python bench_blackhole.py \
		--out $(OUT) --layers $(LAYERS) --sizes $(SIZES) --profile

clean:
	rm -rf $(BUILD_DIR) $(SFPU_BUILD_DIR) $(IP_BUILD_DIR) $(INT8_BUILD_DIR)
