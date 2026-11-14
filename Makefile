.PHONY: help install clean data yolo labels splits feats e1 e2 e3 e4 e5 reproduce all

export PYTHONPATH := .
PYTHON ?= python3

help:
	@echo "Automated Pothole Detection & Infrastructure Audit Pipeline"
	@echo "Targets:"
	@echo "  make install     Install all dependencies"
	@echo "  make data        Download datasets (Czech, India, US, Japan)"
	@echo "  make yolo        Convert annotations from VOC XML to YOLO format"
	@echo "  make labels      Generate labels.csv metadata & audit summary"
	@echo "  make splits      Create group-aware train/val/test splits"
	@echo "  make feats       Extract and cache Color, HOG, LBP, GLCM features"
	@echo "  make e1          Run Experiment 1 (Classical Models Benchmark)"
	@echo "  make e2          Run Experiment 2 (Feature Ablations)"
	@echo "  make e3          Run Experiment 3 (Cross-Domain Generalization)"
	@echo "  make e4          Run Experiment 4 (YOLOv8 Detection & Severity Audit)"
	@echo "  make e5          Run Experiment 5 (Random Forest vs ResNet-18 Benchmark)"
	@echo "  make reproduce   Execute master reproduction script to generate all results"
	@echo "  make all         Execute complete end-to-end pipeline"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/data/download.py

yolo:
	$(PYTHON) src/data/voc2yolo.py

labels:
	$(PYTHON) src/data/labels.py

splits:
	$(PYTHON) src/data/splits.py

feats:
	$(PYTHON) src/feats/build.py

e1:
	$(PYTHON) experiments/e1_models.py

e2:
	$(PYTHON) experiments/e2_feats.py

e3:
	$(PYTHON) experiments/e3_cross.py

e4:
	$(PYTHON) experiments/e4_audit.py --train

failure:
	$(PYTHON) experiments/failure_cases.py

e5:
	$(PYTHON) experiments/compare_rf_vs_cnn.py

reproduce:
	$(PYTHON) reproduce_all_results.py

all: data yolo labels splits feats e1 e2 e3 failure e4 e5 reproduce
	@echo "Complete pothole audit pipeline finished successfully!"

clean:
	rm -rf data/interim/features/*.npy
	rm -rf __pycache__ src/*/__pycache__ experiments/__pycache__
