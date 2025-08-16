#!/usr/bin/env python3
"""
ESS和ERT性能测试脚本
测试不同配置下的渲染性能和质量
"""

import os
import sys
import time
import torch
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import cfg, make_cfg
from src.datasets import make_data_loader
from src.models import make_network
from src.models.nerf.renderer.volume_renderer import Renderer


class ESSERTTester:
    def __init__(self, config_path="configs/nerf/lego.yaml"):
        """初始化测试器"""
        self.config_path = config_path
        self.results = {}
        
    def load_model_and_data(self):
        """加载训练好的模型和测试数据"""
        print("Loading model and data...")
        
        # 加载配置
        make_cfg(self.config_path)
        
        # 加载模型
        self.network = make_network(cfg)
        
        # 加载训练好的权重（如果存在）
        model_path = "data/trained_model/nerf_replication/lego/latest.pth"
        if os.path.exists(model_path):
            print(f"Loading pretrained model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            self.network.load_state_dict(checkpoint['net'])
        else:
            print("Warning: No pretrained model found. Using random weights.")
        
        self.network.eval()
        
        # 加载测试数据
        self.test_loader = make_data_loader(cfg, split='test')
        
        print(f"Loaded model with {sum(p.numel() for p in self.network.parameters())} parameters")
        
    def test_configuration(self, enable_ess=True, enable_ert=True, test_frames=5):
        """测试特定配置的性能"""
        print(f"\n=== Testing ESS={enable_ess}, ERT={enable_ert} ===")
        
        # 临时修改配置
        original_ess = cfg.enable_ess if hasattr(cfg, 'enable_ess') else True
        original_ert = cfg.enable_ert if hasattr(cfg, 'enable_ert') else True
        
        cfg.enable_ess = enable_ess
        cfg.enable_ert = enable_ert
        
        # 创建渲染器
        renderer = Renderer(self.network)
        
        # 性能统计
        render_times = []
        frame_count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if frame_count >= test_frames:
                    break
                    
                print(f"Rendering frame {frame_count + 1}/{test_frames}...")
                
                # 开始计时
                start_time = time.time()
                
                try:
                    # 渲染
                    ret = renderer.render(batch)
                    
                    # 等待GPU完成
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # 结束计时
                    end_time = time.time()
                    render_time = end_time - start_time
                    render_times.append(render_time)
                    
                    print(f"  Render time: {render_time:.4f}s")
                    
                    # 检查输出质量
                    if 'rgb_map' in ret:
                        rgb = ret['rgb_map']
                    else:
                        rgb = ret['rgb_map_0']
                    
                    rgb_np = rgb.cpu().numpy()
                    print(f"  RGB range: [{rgb_np.min():.4f}, {rgb_np.max():.4f}]")
                    print(f"  RGB mean: {rgb_np.mean():.4f}")
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"Error rendering frame {frame_count}: {e}")
                    continue
        
        # 恢复原始配置
        cfg.enable_ess = original_ess
        cfg.enable_ert = original_ert
        
        # 计算统计信息
        if render_times:
            avg_time = np.mean(render_times)
            std_time = np.std(render_times)
            total_time = np.sum(render_times)
            
            result = {
                'avg_time': avg_time,
                'std_time': std_time,
                'total_time': total_time,
                'frames': len(render_times),
                'fps': len(render_times) / total_time if total_time > 0 else 0
            }
            
            print(f"Results:")
            print(f"  Average time per frame: {avg_time:.4f} ± {std_time:.4f}s")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  FPS: {result['fps']:.2f}")
            
            return result
        else:
            print("No successful renders")
            return None
    
    def run_performance_comparison(self):
        """运行完整的性能比较测试"""
        print("=== ESS & ERT Performance Comparison ===")
        
        # 测试不同配置
        configs = [
            (False, False, "Baseline (no optimizations)"),
            (True, False, "ESS only"), 
            (False, True, "ERT only"),
            (True, True, "ESS + ERT")
        ]
        
        results = {}
        
        for enable_ess, enable_ert, label in configs:
            result = self.test_configuration(enable_ess, enable_ert, test_frames=3)
            if result:
                results[label] = result
        
        # 打印比较结果
        print("\n=== Performance Comparison Summary ===")
        baseline = results.get("Baseline (no optimizations)")
        
        for label, result in results.items():
            print(f"\n{label}:")
            print(f"  Average time: {result['avg_time']:.4f}s")
            print(f"  FPS: {result['fps']:.2f}")
            
            if baseline and label != "Baseline (no optimizations)":
                speedup = baseline['avg_time'] / result['avg_time']
                fps_improvement = (result['fps'] - baseline['fps']) / baseline['fps'] * 100
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  FPS improvement: {fps_improvement:+.1f}%")
        
        return results
    
    def test_ess_effectiveness(self):
        """测试ESS空空间跳过的有效性"""
        print("\n=== ESS Effectiveness Test ===")
        
        cfg.enable_ess = True
        cfg.enable_ert = False
        
        renderer = Renderer(self.network)
        
        # 测试一帧
        batch = next(iter(self.test_loader))
        
        with torch.no_grad():
            # 初始化占用网格
            if renderer.occupancy_grid is None:
                renderer._initialize_occupancy_grid()
            
            print(f"Occupancy grid shape: {renderer.occupancy_grid.shape}")
            print(f"Occupancy grid resolution: {renderer.occupancy_grid_resolution}")
            
            # 计算占用率
            occupied_voxels = renderer.occupancy_grid.sum().item()
            total_voxels = renderer.occupancy_grid.numel()
            occupancy_rate = occupied_voxels / total_voxels
            
            print(f"Occupied voxels: {occupied_voxels}/{total_voxels}")
            print(f"Occupancy rate: {occupancy_rate:.4f} ({occupancy_rate*100:.2f}%)")
            print(f"Empty space ratio: {(1-occupancy_rate)*100:.2f}%")
            
            # 渲染并统计ESS效果
            ret = renderer.render(batch)
            
            # 这里可以添加更详细的ESS统计
            print("ESS test completed")
    
    def test_ert_effectiveness(self):
        """测试ERT早期射线终止的有效性"""
        print("\n=== ERT Effectiveness Test ===")
        
        cfg.enable_ess = False
        cfg.enable_ert = True
        
        # 测试不同的ERT阈值
        thresholds = [0.001, 0.01, 0.1]
        
        for threshold in thresholds:
            print(f"\nTesting ERT threshold: {threshold}")
            
            cfg.ert_threshold = threshold
            renderer = Renderer(self.network)
            
            batch = next(iter(self.test_loader))
            
            start_time = time.time()
            with torch.no_grad():
                ret = renderer.render(batch)
            end_time = time.time()
            
            render_time = end_time - start_time
            print(f"  Render time: {render_time:.4f}s")
            
            # 检查渲染质量
            if 'rgb_map' in ret:
                rgb = ret['rgb_map']
            else:
                rgb = ret['rgb_map_0']
                
            rgb_np = rgb.cpu().numpy()
            print(f"  RGB quality - mean: {rgb_np.mean():.4f}, std: {rgb_np.std():.4f}")
    
    def run_full_test(self):
        """运行完整测试套件"""
        print("Starting ESS & ERT comprehensive test...")
        
        try:
            self.load_model_and_data()
            
            # 1. 性能比较测试
            performance_results = self.run_performance_comparison()
            
            # 2. ESS有效性测试
            self.test_ess_effectiveness()
            
            # 3. ERT有效性测试
            self.test_ert_effectiveness()
            
            print("\n=== Test Summary ===")
            print("All tests completed successfully!")
            
            if performance_results:
                print("Performance improvements:")
                baseline = performance_results.get("Baseline (no optimizations)")
                best_config = performance_results.get("ESS + ERT")
                
                if baseline and best_config:
                    overall_speedup = baseline['avg_time'] / best_config['avg_time']
                    print(f"  Best speedup achieved: {overall_speedup:.2f}x")
                    print(f"  Best FPS: {best_config['fps']:.2f}")
                
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    tester = ESSERTTester()
    tester.run_full_test()


if __name__ == "__main__":
    main()
