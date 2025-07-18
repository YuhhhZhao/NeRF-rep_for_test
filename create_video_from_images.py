#!/usr/bin/env python3
"""
从已渲染的图片创建视频的独立脚本

用法:
    python create_video_from_images.py --image_dir data/result/images --output output/video.mp4 --fps 24
    python create_video_from_images.py --result_dir data/result --output output/comparison.mp4 --type comparison
"""

import argparse
import os
import sys
import glob
import cv2
import imageio
import numpy as np
from tqdm import tqdm

# 不导入 src.config 以避免参数冲突


def create_video_from_images(image_dir, output_video_path, fps=24, pattern='*.png', sort_key=None):
    """
    根据已渲染的图片创建视频
    
    Args:
        image_dir: 包含图片的目录
        output_video_path: 输出视频文件路径
        fps: 帧率
        pattern: 图片文件匹配模式
        sort_key: 排序键函数 (默认按文件名排序)
    """
    # 查找所有图片文件
    image_pattern = os.path.join(image_dir, pattern)
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"No images found in {image_dir} with pattern {pattern}")
        return
    
    # 排序图片文件
    if sort_key is None:
        # 默认按文件名排序
        image_files.sort()
    else:
        image_files.sort(key=sort_key)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"First image: {os.path.basename(image_files[0])}")
    print(f"Last image: {os.path.basename(image_files[-1])}")
    
    # 读取图片并转换为视频格式
    frames = []
    
    for i, img_path in enumerate(tqdm(image_files, desc="Loading images")):
        try:
            # 使用cv2读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # 转换BGR到RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
            
            # 每100帧打印一次进度
            if i % 100 == 0:
                print(f"Loaded {i+1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    if not frames:
        print("No valid images found")
        return
    
    print(f"Loaded {len(frames)} valid images")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存视频
    try:
        print(f"Creating video: {output_video_path}")
        print(f"Video settings: {len(frames)} frames, {fps} fps")
        
        imageio.mimwrite(
            output_video_path,
            frames,
            fps=fps,
            quality=8,
            macro_block_size=16
        )
        
        print(f"Video saved successfully: {output_video_path}")
        duration = len(frames) / fps
        print(f"Video duration: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error creating video: {e}")
        import traceback
        traceback.print_exc()


def create_comparison_video(images_dir, output_video_path, fps=24):
    """
    创建预测和真实图片的对比视频
    
    Args:
        images_dir: 图片目录
        output_video_path: 输出视频路径
        fps: 帧率
    """
    # 查找预测和真实图片
    pred_files = glob.glob(os.path.join(images_dir, '*_pred.png'))
    gt_files = glob.glob(os.path.join(images_dir, '*_gt.png'))
    
    if not pred_files or not gt_files:
        print(f"Not enough images found in {images_dir}")
        print(f"Found {len(pred_files)} pred files and {len(gt_files)} gt files")
        return
    
    # 按索引排序
    pred_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', '')))
    gt_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', '')))
    
    print(f"Creating comparison video with {len(pred_files)} frames")
    
    frames = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        try:
            # 读取图片
            pred_img = cv2.imread(pred_file)
            gt_img = cv2.imread(gt_file)
            
            if pred_img is None or gt_img is None:
                continue
            
            # 转换BGR到RGB
            pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            # 水平拼接图片 (左边预测，右边真实)
            combined = np.hstack([pred_rgb, gt_rgb])
            frames.append(combined)
            
        except Exception as e:
            print(f"Error processing {pred_file} and {gt_file}: {e}")
            continue
    
    if not frames:
        print("No valid image pairs found")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存对比视频
    try:
        print(f"Creating comparison video: {output_video_path}")
        
        imageio.mimwrite(
            output_video_path,
            frames,
            fps=fps,
            quality=8,
            macro_block_size=16
        )
        
        print(f"Comparison video saved: {output_video_path}")
        
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Create videos from rendered images')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--result_dir', type=str, help='Result directory (for evaluation images)')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--fps', type=int, default=24, help='Video frame rate')
    parser.add_argument('--pattern', type=str, default='*.png', help='Image file pattern')
    parser.add_argument('--type', type=str, default='pred', choices=['pred', 'gt', 'comparison', 'custom'],
                       help='Video type: pred (prediction), gt (ground truth), comparison (side-by-side), custom (custom pattern)')
    
    args = parser.parse_args()
    
    if args.type == 'comparison':
        # 创建对比视频
        if args.result_dir:
            images_dir = os.path.join(args.result_dir, 'images')
        elif args.image_dir:
            images_dir = args.image_dir
        else:
            print("Error: Need either --result_dir or --image_dir for comparison video")
            return
        
        create_comparison_video(images_dir, args.output, args.fps)
        
    elif args.type in ['pred', 'gt']:
        # 创建预测或真实图片的视频
        if args.result_dir:
            images_dir = os.path.join(args.result_dir, 'images')
        elif args.image_dir:
            images_dir = args.image_dir
        else:
            print("Error: Need either --result_dir or --image_dir")
            return
        
        if args.type == 'pred':
            pattern = '*_pred.png'
        else:
            pattern = '*_gt.png'
        
        create_video_from_images(
            images_dir,
            args.output,
            fps=args.fps,
            pattern=pattern,
            sort_key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', ''))
        )
        
    elif args.type == 'custom':
        # 使用自定义模式
        if not args.image_dir:
            print("Error: Need --image_dir for custom video")
            return
        
        create_video_from_images(
            args.image_dir,
            args.output,
            fps=args.fps,
            pattern=args.pattern
        )
    
    print("Video creation completed!")


if __name__ == '__main__':
    main()
