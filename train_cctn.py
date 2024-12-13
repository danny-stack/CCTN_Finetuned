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
    """设置日志级别"""
    # 设置 root logger 的级别
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置所有 mmcv 和 mmdet 相关 logger 的级别
    for name in ['mmcv', 'mmdet', 'mmseg', 'mmdet.apis.train']:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        # 移除所有处理器
        logger.handlers = []
        # 添加一个只处理 ERROR 及以上级别的处理器
        handler = logging.StreamHandler()
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
#warnings.filterwarnings('ignore', message='.*On January 1, 2023, MMCV.*')

# def main():
#     warnings.filterwarnings('ignore', message='.*On January 1, 2023, MMCV.*')

#     # 加载配置
#     cfg = Config.fromfile('configs/cascade_mask_rcnn_hrnetv2p_w32_20e_v2_finetune.py')
    
#     # 修改配置
#     cfg.work_dir = './work_dirs/cctn_finetune'
#     cfg.device = get_device()
#     cfg.gpu_ids = [0]
    
#     # 构建数据集
#     datasets = [build_dataset(cfg.data.train)]
    
#     # 构建模型
#     model = build_detector(
#         cfg.model,
#         train_cfg=cfg.get('train_cfg'),
#         test_cfg=cfg.get('test_cfg'))
    
#     # 加载预训练权重
#     load_checkpoint(model, 'epoch_36_v2.pth', map_location='cpu')
    
#     # 开始训练
#     train_detector(
#         model,
#         datasets,
#         cfg,
#         distributed=False,
#         validate=True,
#         timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
#         meta=dict()
#     )

def main():
    warnings.filterwarnings('ignore')
    #logging.getLogger('mmcv').setLevel(logging.ERROR)
    setup_logging()
    
    # 加载配置
    #cfg = Config.fromfile('configs/cascade_mask_rcnn_hrnetv2p_w32_20e_v2_finetune.py')
    cfg = Config.fromfile('configs/config_v2.py')
    
    # 设置GPU
    # cfg.gpu_ids = [0]  
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cfg.device = device
    os.makedirs(cfg.work_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 使用第一个GPU
        cfg.gpu_ids = range(1)  # 只使用一个GPU
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'
        cfg.gpu_ids = []
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 构建模型
    model = build_detector(cfg.model)
    model.init_weights()
    model = model.to(cfg.device)
    
    # 加载预训练权重
    checkpoint = load_checkpoint(model, './checkpoints/epoch_20.pth', map_location='cpu')
    
    # 开始训练
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