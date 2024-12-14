from mmdet.apis import init_detector, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config
from mmcv.runner import load_checkpoint
import time
import warnings
import torch
import os.path as osp
import os
import logging

def setup_logging():
    logging.getLogger().setLevel(logging.ERROR)
    
    for name in ['mmcv', 'mmdet', 'mmseg', 'mmdet.apis.train']:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.handlers = []
        handler = logging.StreamHandler()
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
        #warnings.filterwarnings('ignore', message='.*On January 1, 2023, MMCV.*')


def main():
    warnings.filterwarnings('ignore')
    #logging.getLogger('mmcv').setLevel(logging.ERROR)
    setup_logging()
    
    cfg = Config.fromfile('configs/config_v2.py')
    os.makedirs(cfg.work_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  
        cfg.gpu_ids = range(1)  # The first GPU only
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'
        cfg.gpu_ids = []
    

    datasets = [build_dataset(cfg.data.train)]
    
    model = build_detector(cfg.model)
    model.init_weights()
    model = model.to(cfg.device)
    
    # checkpoint = load_checkpoint(model, './checkpoints/epoch_20.pth', map_location='cpu')
    checkpoint = load_checkpoint(model, './checkpoints/best_bbox_mAP_epoch_13.pth', map_location='cpu')
    
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict()
    )

if __name__ == '__main__':
    main()