#!/usr/bin/env python3
"""
专门用于渲染新视角序列的脚本

用法:
    python render_novel_views.py --cfg_file configs/nerf/lego.yaml --render_type spiral
    python render_novel_views.py --cfg_file configs/nerf/lego.yaml --render_type original
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import cfg
from src.utils.net_utils import load_network
from src.datasets.make_dataset import make_dataset


def render_novel_views(cfg_file, render_type='spiral', output_dir=None, render_num=None):
    """
    渲染新视角序列
    
    Args:
        cfg_file: 配置文件路径
        render_type: 渲染类型 ('spiral' 或 'original')
        output_dir: 输出目录 (可选)
        render_num: 渲染帧数 (可选)
    """
    # 加载配置
    cfg.from_file(cfg_file)
    
    # 覆盖参数
    if render_num is not None:
        cfg.render_num = render_num
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(cfg.result_dir, "novel_views")
    
    print(f"Configuration loaded from: {cfg_file}")
    print(f"Render type: {render_type}")
    print(f"Output directory: {output_dir}")
    print(f"Number of frames: {cfg.render_num}")
    
    # 创建数据集
    dataset = make_dataset(cfg.test_dataset_module, cfg.test_dataset_path, cfg.test_dataset_cfg)
    
    # 加载模型
    network = load_network(cfg.network_module, cfg.network_path, cfg.network_cfg)
    network.eval()
    
    # 创建渲染器
    renderer_module = __import__(cfg.renderer_module, fromlist=[''])
    renderer = renderer_module.Renderer(network)
    
    # 获取数据集参数
    poses = dataset.poses  # [N, 4, 4]
    H, W = dataset.H, dataset.W
    focal = dataset.focal
    hwf = [H, W, focal]
    
    # 构造相机内参
    intrinsics = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 渲染新视角序列
    print(f"\nStarting novel view rendering...")
    images_dir, video_path = renderer.render_novel_view_sequence(
        poses=poses,
        hwf=hwf,
        output_dir=output_dir,
        exp_name=f"{cfg.exp_name}_{render_type}",
        iteration=0,
        intrinsics=intrinsics,
        render_type=render_type
    )
    
    print(f"\n✅ Novel view rendering completed!")
    print(f"📁 Images saved to: {images_dir}")
    print(f"🎬 Video saved to: {video_path}")
    
    return images_dir, video_path


def main():
    parser = argparse.ArgumentParser(description='Render novel view sequences')
    parser.add_argument('--cfg_file', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--render_type', type=str, default='spiral', 
                       choices=['spiral', 'original'],
                       help='Type of camera path: spiral (default) or original')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: cfg.result_dir/novel_views)')
    parser.add_argument('--render_num', type=int, default=None,
                       help='Number of frames to render (default: from config)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.cfg_file):
        print(f"Error: Configuration file not found: {args.cfg_file}")
        return
    
    # 渲染新视角序列
    try:
        images_dir, video_path = render_novel_views(
            cfg_file=args.cfg_file,
            render_type=args.render_type,
            output_dir=args.output_dir,
            render_num=args.render_num
        )
        
        print(f"\n🎉 Success! Check the output:")
        print(f"   Images: {images_dir}")
        print(f"   Video: {video_path}")
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
