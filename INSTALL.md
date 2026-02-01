# Installation Guide

## Python 3.14 Users (Current Limitation)

You're using **Python 3.14**, which is very new. Some production ML libraries don't have wheels yet.

### ✅ What Works
- Training models
- Evaluation 
- All metrics and visualizations
- MLflow experiment tracking
- PyTorch model saving

### ❌ What Doesn't Work (Yet)
- ONNX export (requires `onnxruntime`)
- FastAPI serving (requires `pydantic` v2)

### Quick Install

```powershell
# Install training dependencies only
pip install -r requirements-train.txt
```

### Expected Output
You may see some warnings, but the core training libraries should install successfully.

---

## Recommended: Use Python 3.11 or 3.12

For **full production deployment** capabilities:

### Option 1: Install Python 3.11

1. Download from: https://www.python.org/downloads/
2. Install Python 3.11.x
3. Create virtual environment:
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: Use Docker (Enterprise Standard)

This is how top-tier companies deploy ML models:

```powershell
# Build training container
docker build -t swarm-ids-train -f docker/Dockerfile.train .

# Run training
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models swarm-ids-train

# Build serving container (with ONNX)
docker build -t swarm-ids-serve -f docker/Dockerfile.serve .

# Run server
docker run -p 8000:8000 swarm-ids-serve
```

---

## Verify Installation

```powershell
# Check Python version
python --version

# Test imports
python -c "import torch; import pandas; import sklearn; print('✓ Core libraries OK')"

# Run unit tests
python -m pytest tests/ -v
```

---

## Next Steps

Once installed:

1. **Download CICIDS2017 Dataset**
   - Place CSV files in `data/raw/`
   - Files: Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv, etc.

2. **Start Training**
   ```powershell
   python scripts/train.py --epochs 50 --batch-size 256
   ```

3. **Evaluate Model**
   ```powershell
   python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth
   ```

4. **View Results**
   - Training metrics: `mlflow ui` → http://localhost:5000
   - Evaluation plots: `evaluation_results/`

---

## Troubleshooting

### Error: "No module named 'onnxruntime'"
**Solution:** You're on Python 3.14. Either:
- Skip ONNX export for now (training still works)
- Use Python 3.11/3.12
- Use Docker

### Error: "CUDA not available"
**Solution:** CPU training works fine, just slower. To use GPU:
1. Install CUDA toolkit
2. Install GPU version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Error: "MLflow connection refused"
**Solution:** Start MLflow server first:
```powershell
mlflow ui --port 5000
```

Or disable MLflow:
```powershell
python scripts/train.py --no-mlflow
```
