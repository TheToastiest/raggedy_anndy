# Makefile
.PHONY: default-freeze freeze-bin

SHELL := /usr/bin/env bash
FREEZE_BIN := ./target/release/freeze

# Tunables (override on the CLI, e.g.: make default-freeze NPROBE=920)
BACKEND    ?= ivf-pq
N          ?= 20000
DIM        ?= 256
METRIC     ?= cosine
K          ?= 5
NLIST      ?= 1536
NPROBE     ?= 906
REFINE     ?= 400
M          ?= 128
NBITS      ?= 8
ITERS      ?= 60
OPQ_MODE   ?= pca-perm
OPQ_SWEEPS ?= 8
QUERIES    ?= 200
WARMUP     ?= 5
SEED_DATA  ?= 42
SEED_QUER  ?= 999
SEED_KM    ?= 7
MIN_LB95   ?= 0.90
MIN_REC    ?= 0.90
MAX_P95_MS ?= 40
THREADS    ?= 2
INTRA      ?= 8

ARGS := \
  --backend $(BACKEND) \
  --n $(N) --dim $(DIM) --metric $(METRIC) --k $(K) \
  --nlist $(NLIST) --nprobe $(NPROBE) --refine $(REFINE) \
  --m $(M) --nbits $(NBITS) --iters $(ITERS) \
  --opq --opq-mode $(OPQ_MODE) --opq-sweeps $(OPQ_SWEEPS) \
  --queries $(QUERIES) --warmup $(WARMUP) \
  --seed-data $(SEED_DATA) --seed-queries $(SEED_QUER) --seed-kmeans $(SEED_KM) \
  --min-lb95 $(MIN_LB95) --min-recall $(MIN_REC) --max-p95-ms $(MAX_P95_MS) \
  --threads $(THREADS) --intra $(INTRA)

freeze-bin:
	cargo build --release --bin freeze

default-freeze: freeze-bin
	$(FREEZE_BIN) $(ARGS)
