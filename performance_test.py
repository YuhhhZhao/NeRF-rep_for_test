#!/usr/bin/env python3
"""
性能监控工具：比较启用/禁用ESS和ERT优化的渲染性能
"""

import time
import torch
import numpy as np
import os
import yaml


class PerformanceMonitor:
    def __init__(self):
        self.stats = {
            'render_times': [],
            'samples_per_ray': [],
            'effective_samples': [],
            'ess_savings': [],
            'ert_savings': []
        }
    
    def start_timer(self):
        self.start_time = time.time()
    
    def end_timer(self, label=""):
        end_time = time.time()
        duration = end_time - self.start_time
        self.stats['render_times'].append(duration)
        if label:
            print(f"{label}: {duration:.4f} seconds")
        return duration
    
    def log_sampling_stats(self, original_samples, effective_samples, ess_skipped=0, ert_terminated=0):
        """记录采样统计信息"""
        self.stats['samples_per_ray'].append(original_samples)
        self.stats['effective_samples'].append(effective_samples)
        
        if original_samples > 0:
            ess_saving = ess_skipped / original_samples
            ert_saving = ert_terminated / original_samples
            self.stats['ess_savings'].append(ess_saving)
            self.stats['ert_savings'].append(ert_saving)
    
    def get_summary(self):
        """获取性能总结"""
        if not self.stats['render_times']:
            return "No performance data recorded"
        
        avg_time = np.mean(self.stats['render_times'])
        min_time = np.min(self.stats['render_times'])
        max_time = np.max(self.stats['render_times'])
        
        summary = f"""
Performance Summary:
==================
Average render time: {avg_time:.4f}s
Min render time: {min_time:.4f}s
Max render time: {max_time:.4f}s
Total renders: {len(self.stats['render_times'])}
        """
        
        if self.stats['ess_savings']:
            avg_ess_saving = np.mean(self.stats['ess_savings'])
            avg_ert_saving = np.mean(self.stats['ert_savings'])
            
            summary += f"""
ESS (Empty Space Skipping):
  Average samples saved: {avg_ess_saving:.1%}
  
ERT (Early Ray Termination):
  Average samples saved: {avg_ert_saving:.1%}
  
Combined speedup potential: {(avg_ess_saving + avg_ert_saving):.1%}
            """
        
        return summary


def run_performance_test():
    """运行性能测试"""
    print("Starting NeRF Performance Test with ESS/ERT optimizations")
    print("="*60)
    
    # 配置文件路径
    config_path = "configs/nerf/lego.yaml"
    
    # 测试配置：启用/禁用优化
    test_configs = [
        {"enable_ess": False, "enable_ert": False, "name": "Baseline (No optimization)"},
        {"enable_ess": True, "enable_ert": False, "name": "ESS only"},
        {"enable_ess": False, "enable_ert": True, "name": "ERT only"},
        {"enable_ess": True, "enable_ert": True, "name": "ESS + ERT"},
    ]
    
    results = {}
    
    for test_config in test_configs:
        print(f"\n{'-'*40}")
        print(f"Testing: {test_config['name']}")
        print(f"{'-'*40}")
        
        # 更新配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['enable_ess'] = test_config['enable_ess']
        config['enable_ert'] = test_config['enable_ert']
        
        # 保存临时配置
        temp_config_path = f"temp_config_{test_config['name'].replace(' ', '_')}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # 运行评估（仅测试少量图像以节省时间）
        try:
            import subprocess
            import sys
            
            # 限制测试图像数量
            config['test_dataset']['cams'] = [0, 4, 1]  # 只测试前5张图像
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Running evaluation with {test_config['name']}...")
            
            start_time = time.time()
            
            cmd = [
                sys.executable, "run.py",
                "--cfg_file", temp_config_path,
                "--type", "evaluate"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✓ Test completed successfully in {total_time:.2f} seconds")
                
                # 解析输出以获取渲染统计信息
                output_lines = result.stdout.split('\n')
                render_times = []
                
                for line in output_lines:
                    if 'Image' in line and 'PSNR=' in line:
                        # 简单的时间估算
                        render_times.append(total_time / 5)  # 平均分配到5张图像
                
                results[test_config['name']] = {
                    'total_time': total_time,
                    'avg_time_per_image': total_time / 5 if render_times else total_time,
                    'success': True
                }
                
            else:
                print(f"✗ Test failed with return code: {result.returncode}")
                results[test_config['name']] = {
                    'success': False,
                    'error': result.stderr
                }
            
        except subprocess.TimeoutExpired:
            print(f"✗ Test timed out")
            results[test_config['name']] = {'success': False, 'error': 'Timeout'}
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_config['name']] = {'success': False, 'error': str(e)}
        finally:
            # 清理临时配置文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    # 输出性能比较结果
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*60}")
    
    baseline_time = None
    for name, result in results.items():
        if result.get('success', False):
            total_time = result['total_time']
            avg_time = result['avg_time_per_image']
            
            print(f"\n{name}:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg time per image: {avg_time:.3f}s")
            
            if 'Baseline' in name:
                baseline_time = total_time
            elif baseline_time is not None:
                speedup = baseline_time / total_time
                print(f"  Speedup vs baseline: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
        else:
            print(f"\n{name}: FAILED - {result.get('error', 'Unknown error')}")
    
    # 保存结果
    with open("performance_test_results.txt", 'w') as f:
        f.write("NeRF ESS/ERT Performance Test Results\n")
        f.write("="*40 + "\n\n")
        
        for name, result in results.items():
            f.write(f"{name}:\n")
            if result.get('success', False):
                f.write(f"  Total time: {result['total_time']:.2f}s\n")
                f.write(f"  Avg time per image: {result['avg_time_per_image']:.3f}s\n")
            else:
                f.write(f"  FAILED: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    print(f"\nResults saved to: performance_test_results.txt")


if __name__ == "__main__":
    run_performance_test()
