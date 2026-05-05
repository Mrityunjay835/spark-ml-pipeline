PYTHON := python
PROJECT := spark-ml-project

# ─── Full Pipeline ────────────────────────────────────────────────────────────
all:
	$(PYTHON) main.py --run all --skip eda

all-with-eda:
	$(PYTHON) main.py --run all

# ─── Individual Stages ────────────────────────────────────────────────────────
ingest:
	$(PYTHON) main.py --run ingest

join:
	$(PYTHON) main.py --run join

transform:
	$(PYTHON) main.py --run transform

features:
	$(PYTHON) main.py --run features

eda:
	$(PYTHON) main.py --run eda

train:
	$(PYTHON) main.py --run train --classifier gbt

train-cv:
	$(PYTHON) main.py --run train --classifier gbt --cv

train-rf:
	$(PYTHON) main.py --run train --classifier rf

train-lr:
	$(PYTHON) main.py --run train --classifier lr

# ─── Streaming ────────────────────────────────────────────────────────────────
stream:
	$(PYTHON) main.py --run stream --classifier gbt --sink console

stream-parquet:
	$(PYTHON) main.py --run stream --classifier gbt --sink parquet

generate:
	$(PYTHON) -m src.streaming.data_generator --events 50 --interval 15

# ─── Partial Pipelines ────────────────────────────────────────────────────────
from-features:
	$(PYTHON) main.py --run features,train

from-train:
	$(PYTHON) main.py --run train

# ─── Utilities ────────────────────────────────────────────────────────────────
list-stages:
	$(PYTHON) main.py --list-stages

clean-streaming:
	rm -rf data/streaming/input/* data/streaming/checkpoint/*

clean-models:
	rm -rf models/model_* models/spark_pipeline

clean-processed:
	rm -rf data/processed/* data/features/*

clean-all: clean-streaming clean-models clean-processed

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Full pipeline:"
	@echo "  all              Run ingest→join→transform→features→train (skip eda)"
	@echo "  all-with-eda     Run full pipeline including EDA"
	@echo ""
	@echo "Individual stages:"
	@echo "  ingest           Load raw CSVs to processed parquet"
	@echo "  join             Join tables into wide table"
	@echo "  transform        Clean, deduplicate, cap outliers"
	@echo "  features         Build RFM + behavioral feature store"
	@echo "  eda              Run exploratory analysis"
	@echo "  train            Train GBT model"
	@echo "  train-cv         Train GBT with CrossValidator HPO"
	@echo "  train-rf         Train RandomForest"
	@echo "  train-lr         Train Logistic Regression"
	@echo ""
	@echo "Streaming:"
	@echo "  stream           Start streaming predictions (console sink)"
	@echo "  stream-parquet   Start streaming predictions (parquet sink)"
	@echo "  generate         Start synthetic data generator"
	@echo ""
	@echo "Partial runs:"
	@echo "  from-features    Run features + train only"
	@echo "  from-train       Run train only"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-streaming  Clear streaming input + checkpoint"
	@echo "  clean-models     Delete saved models"
	@echo "  clean-all        Clear everything"

.PHONY: all all-with-eda ingest join transform features eda train train-cv \
        train-rf train-lr stream stream-parquet generate from-features \
        from-train list-stages clean-streaming clean-models clean-processed \
        clean-all help