# This API service is to provide an HTTP interface that allows external systems to: 
# Upload image (in base64 format), use the trained CascadeTableNet model to detect tables in images
# Then returns the test result (the bounding box and confidence level)

import os
import base64
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np

app = FastAPI()

# 加载模型
CONFIG_FILE = './configs/config_v2.py'
CHECKPOINT_FILE = './checkpoints/best_bbox_mAP_epoch_13.pth'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

class BBox(BaseModel):
    left: float
    top: float
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float):
        return cls(left=x1, top=y1, width=x2-x1, height=y2-y1)

class ScoredBBox(BaseModel):
    bbox: BBox
    score: float

class TableDetectionRequest(BaseModel):
    image_base64: str

def base64_to_png(base64_string: str) -> str:
    if base64_string.startswith('data:image/png;base64,'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    output_dir = Path.cwd() / "imgs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"image_{len(list(output_dir.glob('*.png')))}.png"
    with open(output_file, 'wb') as f:
        f.write(image_data)
    return str(output_file)

def process_detections(result: list) -> List[ScoredBBox]:
    boxes = []
    if len(result[0]) > 0:
        for det in result[0]:
            box = BBox.from_xyxy(
                x1=float(det[0]), 
                y1=float(det[1]),
                x2=float(det[2]), 
                y2=float(det[3])
            )
            boxes.append(ScoredBBox(bbox=box, score=float(det[4])))
    return boxes

@app.post("/detect_tables")
async def detect_tables(request: TableDetectionRequest):
    try:
        image_path = base64_to_png(request.image_base64)
        img = mmcv.imread(image_path)
        img = mmcv.imresize(img, (1333, 800))
        result = inference_detector(model, img)
        detections = process_detections(result)
        return jsonable_encoder(detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/metrics")
# async def get_metrics():
#     return {
#         "precision": calculate_precision(),
#         "recall": calculate_recall(),
#         "f1_score": calculate_f1_score()
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)