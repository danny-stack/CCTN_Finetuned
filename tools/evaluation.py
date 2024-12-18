# tools/evaluation.py
from mmdet.apis import init_detector, inference_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
from mmcv import Config
import mmcv
import os
from pathlib import Path
import numpy as np

def visualize_results(model, img_path, save_dir, score_thr=0.85):
    """可视化单张图片的检测结果"""
    img = mmcv.imread(img_path)
    img_resized = mmcv.imresize(img, (1333, 800))
    result = inference_detector(model, img_resized)
    
    # 设置类别名称
    model.CLASSES = ['table']
    
    # 生成可视化结果
    out_img = model.show_result(
        img_resized,
        result,
        score_thr=score_thr,
        bbox_color='green',
        text_color='green',
        show=False
    )
    
    # 保存结果
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    mmcv.imwrite(out_img, save_path)

# def visualize_results(model, img_path, save_dir, score_thr=0.85):
#     if hasattr(model, 'module'):
#         model = model.module  # 获取原始模型
        
#     img = mmcv.imread(img_path)
#     orig_h, orig_w = img.shape[:2]
#     img_resized = mmcv.imresize(img, (1333, 800))
#     result = inference_detector(model, img_resized)
    
#     scale_w = orig_w / 1333
#     scale_h = orig_h / 800
    
#     scaled_result = []
#     for class_results in result:
#         scaled_class = []
#         for det in class_results:
#             scaled_det = det.copy()
#             scaled_det[0] *= scale_w
#             scaled_det[1] *= scale_h
#             scaled_det[2] *= scale_w
#             scaled_det[3] *= scale_h
#             scaled_class.append(scaled_det)
#         scaled_result.append(np.array(scaled_class))  # 转换为 numpy 数组
    
#     model.CLASSES = ['table']
#     out_img = model.show_result(
#         img,
#         scaled_result,
#         score_thr=score_thr,
#         bbox_color='green',
#         text_color='green',
#         show=False
#     )
    
#     save_path = os.path.join(save_dir, os.path.basename(img_path))
#     mmcv.imwrite(out_img, save_path)

def main():
    # 配置
    cfg = Config.fromfile('./configs/config_v2.py')
    
    # 更新测试集路径
    cfg.data.test.ann_file = './CCTN_dataset/annotations/test.json'
    cfg.data.test.img_prefix = './CCTN_dataset/test'
    
    # 创建保存可视化结果的目录
    vis_dir = './work_dirs/evaluation/visualization'
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
    
    # 加载模型
    model = init_detector(cfg, './checkpoints/best_bbox_mAP_epoch_13.pth', device='cuda:0')
    model = MMDataParallel(model, device_ids=[0])
    
    # 执行评估
    outputs = single_gpu_test(model, data_loader)
    
    # 计算评估指标
    eval_results = dataset.evaluate(outputs)
    
    # 打印结果
    print("\n=== Evaluation Metrics ===")
    for metric_name, metric_value in eval_results.items():
        print(f'{metric_name}: {metric_value}')
    
    # 可视化结果
    print("\n=== Generating Visualizations ===")
    for img_info in dataset.data_infos:
        img_path = os.path.join(dataset.img_prefix, img_info['file_name'])
        visualize_results(model.module, img_path, vis_dir)
    print(f"Visualizations saved to {vis_dir}")

if __name__ == '__main__':
    main()