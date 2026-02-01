# Swarm IDS - Production ML Pipeline

🚀 **Enterprise-grade deep learning pipeline for network intrusion detection**

## 🎯 Project Structure (Production-Ready)

```
swarm-ids-ml/
├── cicddata/              # Your real CICIDS2017 dataset (4 CSV files)
├── configs/               # Hydra configuration files
│   └── training/
│       └── base.yaml
├── src/                   # Core source code
│   ├── data/             # Data pipeline
│   │   ├── loader.py           # Data loading & temporal splitting
│   │   ├── preprocessors.py   # Feature engineering (RobustScaler)
│   │   ├── dataset.py          # PyTorch Dataset
│   │   └── __init__.py
│   ├── models/           # Model architectures
│   │   ├── efficientnet.py    # EfficientNet-IDS
│   │   ├── gbdt.py            # LightGBM baseline
│   │   └── __init__.py
│   ├── training/         # Training pipeline
│   │   ├── trainer.py         # Advanced trainer (AMP, early stopping)
│   │   ├── losses.py          # Focal Loss, Label Smoothing
│   │   ├── metrics.py         # Metrics calculation
│   │   └── __init__.py
│   ├── evaluation/       # Evaluation framework
│   │   ├── evaluator.py       # Comprehensive metrics & visualization
│   │   └── __init__.py
│   ├── deployment/       # Production deployment
│   │   ├── onnx_export.py     # ONNX conversion
│   │   ├── serving.py         # FastAPI server
│   │   └── __init__.py
│   └── __init__.py
├── scripts/              # Executable scripts
│   ├── train_production.py   # Production training with K-fold CV
│   ├── hyperparameter_optimization.py  # Automated HPO (Optuna)
│   ├── evaluate.py       # Model evaluation
│   ├── export_onnx.py    # ONNX export
│   └── train.py          # Simple training (for testing)
├── tests/                # Unit tests
│   └── test_data_pipeline.py
├── docker/               # Docker deployment
│   └── README.md
├── models/               # Saved models (created during training)
├── .gitignore           # Git ignore rules
├── requirements.txt     # Core dependencies
├── requirements-train.txt  # Training dependencies (Python 3.14 compatible)
├── README.md            # Main documentation
├── INSTALL.md           # Installation guide
└── PRODUCTION_TRAINING.md  # Production training guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements-train.txt
```

### 2. Production Training (Recommended)
```powershell
python scripts/train_production.py `
    --data-dir cicddata `
    --epochs 50 `
    --batch-size 512 `
    --cv-folds 5 `
    --use-focal-loss
```

**Expected Results:**
- **Mean F1 Score: 90-95%**
- **Training Time: 6-10 hours** (CPU) or **1-2 hours** (GPU)

### 3. Hyperparameter Optimization (Advanced)
```powershell
python scripts/hyperparameter_optimization.py `
    --data-dir cicddata `
    --n-trials 50
```

### 4. Evaluate Model
```powershell
python scripts/evaluate.py `
    --checkpoint models/checkpoints/best_model_cv.pth `
    --data-dir cicddata
```

### 5. Export to ONNX
```powershell
python scripts/export_onnx.py `
    --checkpoint models/checkpoints/best_model_cv.pth `
    --output models/onnx/swarm_ids.onnx
```

## 📊 Production Features

### Data Pipeline
- ✅ **Temporal splitting** (no data leakage)
- ✅ **RobustScaler** (outlier-resistant)
- ✅ **Mutual Information** feature selection
- ✅ **Data quality validation** (missing values, duplicates)
- ✅ **Class imbalance handling** (Focal Loss)

### Training
- ✅ **K-Fold Cross-Validation** (5 folds, stratified)
- ✅ **Mixed Precision Training** (AMP for 2x speedup)
- ✅ **Gradient Clipping** (stability)
- ✅ **Cosine Annealing LR** (smooth decay)
- ✅ **Early Stopping** (prevents overfitting)
- ✅ **MLflow Integration** (experiment tracking)

### Model Architecture
- ✅ **EfficientNet-B0** (state-of-the-art)
- ✅ **Squeeze-and-Excitation blocks**
- ✅ **MBConv layers** (mobile inverted bottleneck)
- ✅ **3.7M parameters**

### Evaluation
- ✅ **Comprehensive metrics** (accuracy, precision, recall, F1, ROC AUC)
- ✅ **Per-class analysis**
- ✅ **Confusion matrices** (raw + normalized)
- ✅ **ROC curves** (all attack types)
- ✅ **False Positive Rate** analysis (critical for IDS)

### Deployment
- ✅ **ONNX export** (3-5x speedup, <20ms inference)
- ✅ **FastAPI serving** (RESTful API)
- ✅ **Docker support** (containerized deployment)
- ✅ **Health checks** (production monitoring)

## 📈 Expected Performance

| Metric | Target | Top-Tier |
|--------|--------|----------|
| Accuracy | 92-96% | >95% |
| F1 Score | 90-95% | >93% |
| Benign FPR | 3-6% | <5% |
| Attack Detection Rate | 94-98% | >96% |
| Inference Time | <20ms | <15ms |

## 📚 Documentation

- **[README.md](README.md)** - Project overview
- **[INSTALL.md](INSTALL.md)** - Installation instructions
- **[PRODUCTION_TRAINING.md](PRODUCTION_TRAINING.md)** - Production training guide
- **[docker/README.md](docker/README.md)** - Docker deployment guide

## 🔬 Advanced Features

### Hyperparameter Optimization
Uses **Optuna** for Bayesian optimization:
- Automatically finds best learning rate, batch size, model size
- Prunes poor trials early
- Saves best parameters to JSON

### Cross-Validation
Implements **5-fold stratified cross-validation**:
- Preserves class distribution
- Reports mean ± std dev metrics
- Selects best fold automatically

### Data Quality Checks
Automated validation:
- Missing value imputation
- Infinite value replacement
- Duplicate removal
- Class distribution analysis

## 🛡️ Production-Ready Features

- ✅ **No data leakage** (verified with unit tests)
- ✅ **Type hints** (Python 3.10+ style)
- ✅ **Comprehensive logging** (loguru)
- ✅ **Error handling** (graceful failures)
- ✅ **Modular design** (easy to extend)
- ✅ **Unit tests** (pytest)
- ✅ **Git-ready** (.gitignore configured)

## 🚢 Deployment Options

### Option 1: ONNX + FastAPI (Recommended)
```powershell
# Export model
python scripts/export_onnx.py

# Start server
python -c "from src.deployment import ModelServer; \
    server = ModelServer('models/onnx/swarm_ids.onnx'); \
    server.run()"
```

### Option 2: Docker
```powershell
docker build -t swarm-ids -f docker/Dockerfile.serve .
docker run -p 8000:8000 swarm-ids
```

## 📊 Monitoring

### MLflow UI
```powershell
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Logs
```powershell
Get-Content logs\swarm_ids_*.log -Tail 50 -Wait
```

## 🤝 Contributing

This pipeline follows **industry best practices**:
- Type hints (Google style)
- Docstrings (comprehensive)
- Unit tests (pytest)
- Code organization (modular)
- Git workflow (feature branches)

## 📝 License

MIT License

## 🙏 Acknowledgments

- CICIDS2017 dataset by Canadian Institute for Cybersecurity
- EfficientNet architecture by Google Brain
- Focal Loss by Facebook AI Research

---

**Built for production ML deployments** 🚀

*Questions? Check the documentation or open an issue.*
