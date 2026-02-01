# src/deployment/serving.py
"""
Production model serving with FastAPI.

Features:
- RESTful API
- Batch inference
- Health checks
- Request validation
"""

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, field_validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from typing import List
import numpy as np
from pathlib import Path
import time
from loguru import logger

if FASTAPI_AVAILABLE:
    from .onnx_export import ONNXInference


    class PredictionRequest(BaseModel):
        """Request schema for predictions."""
        
        features: List[List[float]]
        
        @field_validator('features')
        @classmethod
        def validate_features(cls, v):
            if not v:
                raise ValueError("Features cannot be empty")
            if not all(len(row) == len(v[0]) for row in v):
                raise ValueError("All feature vectors must have same length")
            return v


    class PredictionResponse(BaseModel):
        """Response schema for predictions."""
        
        predictions: List[int]
        probabilities: List[List[float]]
        attack_types: List[str]
        confidence_scores: List[float]
        latency_ms: float


    class ModelServer:
        """
        Production model serving application.
        
        Endpoints:
        - POST /predict: Batch predictions
        - GET /health: Health check
        - GET /info: Model information
        """
        
        ATTACK_TYPES = [
            'BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris',
            'DoS Slowhttptest', 'PortScan', 'DDoS', 'FTP-Patator',
            'SSH-Patator', 'Bot', 'Web Attack – Brute Force',
            'Web Attack – XSS', 'Web Attack – Sql Injection',
            'Infiltration', 'Heartbleed'
        ]
        
        def __init__(
            self,
            model_path: Path,
            model_version: str = "1.0.0",
            use_gpu: bool = False
        ):
            self.model_path = Path(model_path)
            self.model_version = model_version
            
            # Load model
            logger.info("Loading ONNX model...")
            self.model = ONNXInference(model_path, use_gpu=use_gpu)
            logger.info("✓ Model loaded successfully")
            
            # Create FastAPI app
            self.app = FastAPI(
                title="Swarm IDS Model Server",
                description="Production ML inference for network intrusion detection",
                version=model_version
            )
            
            # Add CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Register routes
            self._register_routes()
            
        def _register_routes(self):
            """Register API endpoints."""
            
            @self.app.post("/predict", response_model=PredictionResponse)
            async def predict(request: PredictionRequest):
                """Batch prediction endpoint."""
                
                start_time = time.time()
                
                try:
                    # Convert to numpy
                    X = np.array(request.features, dtype=np.float32)
                    
                    # Predict
                    predictions = self.model.predict(X)
                    probabilities = self.model.predict_proba(X)
                    
                    # Get attack types
                    attack_types = [self.ATTACK_TYPES[p] for p in predictions]
                    
                    # Get confidence scores
                    confidence_scores = np.max(probabilities, axis=1).tolist()
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return PredictionResponse(
                        predictions=predictions.tolist(),
                        probabilities=probabilities.tolist(),
                        attack_types=attack_types,
                        confidence_scores=confidence_scores,
                        latency_ms=latency_ms
                    )
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
                    
            @self.app.get("/health")
            async def health():
                """Health check endpoint."""
                return {
                    "status": "healthy",
                    "model_version": self.model_version,
                    "model_path": str(self.model_path)
                }
                
            @self.app.get("/info")
            async def info():
                """Model information endpoint."""
                return {
                    "model_version": self.model_version,
                    "attack_types": self.ATTACK_TYPES,
                    "num_classes": len(self.ATTACK_TYPES),
                    "model_path": str(self.model_path)
                }
                
        def run(self, host: str = "0.0.0.0", port: int = 8000):
            """Run the server."""
            uvicorn.run(self.app, host=host, port=port)

else:
    # Stub implementation if FastAPI not available
    class ModelServer:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
