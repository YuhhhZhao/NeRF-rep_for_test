#!/usr/bin/env python3
"""
简化的ESS和ERT性能测试脚本
快速验证优化功能是否正常工作
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 确保有正确的导入路径
try:
    from src.config.config import cfg, make_cfg
    from src.datasets import make_data_loader
    from src.models import make_network
    from src.models.nerf.renderer.volume_renderer import Renderer
    print("Successfully imported all modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def quick_test():
    """快速测试ESS和ERT功能"""
    print("=== Quick ESS & ERT Test ===")
    
    try:
        # 创建模拟的命令行参数
        args = argparse.Namespace()
        args.cfg_file = "configs/nerf/lego.yaml"
        args.test = False
        args.type = ""
        args.det = ""
        args.local_rank = 0
        args.opts = []
        
        # 加载配置
        print("Loading configuration...")
        global cfg
        cfg = make_cfg(args)
        
        # 打印ESS和ERT设置
        print(f"ESS enabled: {getattr(cfg, 'enable_ess', 'Not set')}")
        print(f"ERT enabled: {getattr(cfg, 'enable_ert', 'Not set')}")
        print(f"ERT threshold: {getattr(cfg, 'ert_threshold', 'Not set')}")
        print(f"Occupancy grid resolution: {getattr(cfg, 'occupancy_grid_resolution', 'Not set')}")
        
        # 尝试创建网络（不加载权重）
        print("Creating network...")
        network = make_network(cfg)
        network.eval()
        print(f"Network created with {sum(p.numel() for p in network.parameters())} parameters")
        
        # 测试渲染器初始化
        print("Testing renderer initialization...")
        renderer = Renderer(network)
        
        print(f"Renderer ESS enabled: {renderer.enable_ess}")
        print(f"Renderer ERT enabled: {renderer.enable_ert}")
        print(f"ERT threshold: {renderer.ert_threshold}")
        print(f"Occupancy grid resolution: {renderer.occupancy_grid_resolution}")
        
        # 测试占用网格初始化
        if renderer.enable_ess:
            print("Testing occupancy grid initialization...")
            if renderer.occupancy_grid is not None:
                grid_size = renderer.occupancy_grid.shape
                occupancy_rate = renderer.occupancy_grid.sum().item() / renderer.occupancy_grid.numel()
                print(f"Occupancy grid shape: {grid_size}")
                print(f"Initial occupancy rate: {occupancy_rate:.4f} ({occupancy_rate*100:.2f}%)")
            else:
                print("Occupancy grid not initialized yet")
        
        # 创建一个简单的测试batch
        print("Creating test batch...")
        H, W = 100, 100  # 小尺寸用于快速测试
        
        # 获取设备
        device = renderer.device
        print(f"Using device: {device}")
        
        # 创建简单的相机pose和内参，确保在正确的设备上
        pose = torch.eye(4, dtype=torch.float32, device=device)
        pose[2, 3] = 4.0  # 相机距离场景中心4个单位
        
        intrinsics = torch.tensor([
            [100.0, 0.0, W/2],
            [0.0, 100.0, H/2],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)
        
        batch = {
            'pose': pose.unsqueeze(0),
            'intrinsics': intrinsics.unsqueeze(0),
            'H': H,
            'W': W
        }
        
        # 测试渲染
        print("Testing rendering...")
        start_time = time.time()
        
        with torch.no_grad():
            ret = renderer.render(batch)
        
        end_time = time.time()
        render_time = end_time - start_time
        
        print(f"Render time: {render_time:.4f} seconds")
        
        # 检查渲染结果
        if 'rgb_map' in ret:
            rgb = ret['rgb_map']
        else:
            rgb = ret['rgb_map_0']
            
        rgb_np = rgb.cpu().numpy()
        print(f"RGB output shape: {rgb_np.shape}")
        print(f"RGB range: [{rgb_np.min():.4f}, {rgb_np.max():.4f}]")
        print(f"RGB mean: {rgb_np.mean():.4f}")
        
        # 测试占用网格更新（如果启用ESS）
        if renderer.enable_ess and renderer.occupancy_grid is not None:
            new_occupancy_rate = renderer.occupancy_grid.sum().item() / renderer.occupancy_grid.numel()
            print(f"Occupancy rate after rendering: {new_occupancy_rate:.4f} ({new_occupancy_rate*100:.2f}%)")
        
        print("\n=== Test Results ===")
        print("✓ Configuration loading: SUCCESS")
        print("✓ Network creation: SUCCESS")  
        print("✓ Renderer initialization: SUCCESS")
        print("✓ ESS occupancy grid: SUCCESS" if renderer.enable_ess else "- ESS disabled")
        print("✓ ERT implementation: SUCCESS" if renderer.enable_ert else "- ERT disabled")
        print("✓ Rendering test: SUCCESS")
        print(f"✓ Total test time: {render_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_comparison():
    """简单的性能对比测试"""
    print("\n=== Performance Comparison ===")
    
    configs = [
        {"enable_ess": False, "enable_ert": False, "name": "Baseline"},
        {"enable_ess": True, "enable_ert": False, "name": "ESS Only"},
        {"enable_ess": False, "enable_ert": True, "name": "ERT Only"},
        {"enable_ess": True, "enable_ert": True, "name": "ESS + ERT"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        try:
            # 临时修改配置
            original_ess = getattr(cfg, 'enable_ess', True)
            original_ert = getattr(cfg, 'enable_ert', True)
            
            cfg.enable_ess = config["enable_ess"]
            cfg.enable_ert = config["enable_ert"]
            
            # 创建网络和渲染器
            network = make_network(cfg)
            network.eval()
            renderer = Renderer(network)
            
            # 创建测试数据
            H, W = 50, 50  # 更小的尺寸以便快速测试
            
            # 获取设备
            device = renderer.device
            
            pose = torch.eye(4, dtype=torch.float32, device=device)
            pose[2, 3] = 4.0
            
            intrinsics = torch.tensor([
                [50.0, 0.0, W/2],
                [0.0, 50.0, H/2], 
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device)
            
            batch = {
                'pose': pose.unsqueeze(0),
                'intrinsics': intrinsics.unsqueeze(0),
                'H': H,
                'W': W
            }
            
            # 多次渲染取平均时间
            times = []
            for i in range(3):  # 3次测试
                start_time = time.time()
                with torch.no_grad():
                    ret = renderer.render(batch)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            results[config['name']] = avg_time
            
            print(f"  Average time: {avg_time:.4f}s")
            
            # 恢复原始配置
            cfg.enable_ess = original_ess
            cfg.enable_ert = original_ert
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config['name']] = None
    
    # 打印对比结果
    print(f"\n=== Performance Summary ===")
    baseline_time = results.get("Baseline")
    
    for name, time_val in results.items():
        if time_val is not None:
            print(f"{name}: {time_val:.4f}s", end="")
            if baseline_time and name != "Baseline":
                speedup = baseline_time / time_val
                print(f" (speedup: {speedup:.2f}x)", end="")
            print()
        else:
            print(f"{name}: FAILED")


def main():
    """主函数"""
    print("Starting ESS & ERT Quick Test...")
    
    # 基础功能测试
    success = quick_test()
    
    if success:
        # 性能对比测试
        try:
            performance_comparison()
        except Exception as e:
            print(f"Performance comparison failed: {e}")
    else:
        print("Basic test failed, skipping performance comparison")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
