import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []
        self.img_count = 0

    def psnr_metric(self, img_pred, img_gt):
        """
        计算PSNR值
        假设输入图像已经归一化到[0,1]范围
        """
        mse = np.mean((img_pred - img_gt) ** 2)
        if mse == 0:
            return float('inf')  # 如果MSE为0，PSNR为无穷大
        
        # 对于归一化到[0,1]的图像，MAX_I = 1
        max_pixel_value = 1.0
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
        # 等价于: psnr = 10 * np.log10(max_pixel_value**2 / mse)
        # 由于max_pixel_value=1，所以: psnr = -10 * np.log10(mse)
        
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        
        # 保存图像（BGR格式用于OpenCV）
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255).astype(np.uint8),
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255).astype(np.uint8),
        )
        
        # 确保图像在[0,1]范围内用于SSIM计算
        img_pred_clipped = np.clip(img_pred, 0, 1)
        img_gt_clipped = np.clip(img_gt, 0, 1)
        
        # 计算SSIM（使用[0,1]范围的float数据）
        try:
            ssim_value = compare_ssim(
                img_pred_clipped, 
                img_gt_clipped, 
                win_size=min(7, min(img_pred.shape[0], img_pred.shape[1])),  # 动态窗口大小
                data_range=1.0,  # 数据范围为0-1
                multichannel=True,
                channel_axis=2
            )
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            # 如果失败，尝试使用uint8格式
            img_pred_uint8 = (img_pred_clipped * 255).astype(np.uint8)
            img_gt_uint8 = (img_gt_clipped * 255).astype(np.uint8)
            ssim_value = compare_ssim(
                img_pred_uint8, 
                img_gt_uint8, 
                win_size=7,
                data_range=255,
                multichannel=True,
                channel_axis=2
            )
        
        return ssim_value

    def evaluate(self, output, batch):
        """
        评估单个批次的输出
        """
        # 从batch中获取真实图像
        if 'image' not in batch:
            print("Warning: No ground truth image found in batch")
            return
        
        # 获取预测的RGB图像
        if 'rgb_map' in output:
            rgb_pred = output['rgb_map']  # 使用fine network的输出
        elif 'rgb_map_0' in output:
            rgb_pred = output['rgb_map_0']  # 使用coarse network的输出
        else:
            print("Warning: No RGB prediction found in output")
            return
        
        # 获取真实图像
        img_gt = batch['image'].squeeze(0)  # [H, W, 3] or [H, W, 4]
        if img_gt.shape[-1] == 4:  # 如果有alpha通道，只取RGB
            img_gt = img_gt[..., :3]
        
        # 确保数据在CPU上且为numpy格式
        if torch.is_tensor(rgb_pred):
            rgb_pred = rgb_pred.detach().cpu().numpy()
        if torch.is_tensor(img_gt):
            img_gt = img_gt.detach().cpu().numpy()
        
        # 获取图像尺寸
        H, W = int(batch['H']), int(batch['W'])
        
        # 重新reshape预测图像为[H, W, 3]
        if rgb_pred.shape[0] == H * W:
            rgb_pred = rgb_pred.reshape(H, W, 3)
        
        # 确保值在[0, 1]范围内
        rgb_pred = np.clip(rgb_pred, 0, 1)
        img_gt = np.clip(img_gt, 0, 1)
        
        # 计算MSE
        mse = np.mean((rgb_pred - img_gt) ** 2)
        self.mse.append(mse)
        
        # 计算PSNR
        psnr = self.psnr_metric(rgb_pred, img_gt)
        self.psnr.append(psnr)
        
        # 计算SSIM并保存图像
        ssim = self.ssim_metric(rgb_pred, img_gt, batch, self.img_count, len(self.imgs))
        self.ssim.append(ssim)
        
        # 保存图像信息
        self.imgs.append({
            'id': self.img_count,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        })
        
        self.img_count += 1
        
        # 打印当前图像的指标
        print(f"Image {self.img_count}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, MSE={mse:.6f}")

    def summarize(self):
        """
        总结所有评估结果
        """
        if len(self.psnr) == 0:
            print("No evaluation results to summarize")
            return
        
        # 计算平均指标
        avg_mse = np.mean(self.mse)
        avg_psnr = np.mean(self.psnr)
        avg_ssim = np.mean(self.ssim)
        
        std_mse = np.std(self.mse)
        std_psnr = np.std(self.psnr)
        std_ssim = np.std(self.ssim)
        
        # 打印结果
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Number of images evaluated: {len(self.psnr)}")
        print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print("="*50)
        
        # 保存详细结果到JSON文件
        result_dir = os.path.join(cfg.result_dir, "metrics")
        os.system("mkdir -p {}".format(result_dir))
        
        # 转换所有图像数据为JSON可序列化的格式
        per_image_results = []
        for img_data in self.imgs:
            per_image_results.append({
                'id': int(img_data['id']),
                'mse': float(img_data['mse']),
                'psnr': float(img_data['psnr']),
                'ssim': float(img_data['ssim'])
            })
        
        results = {
            'summary': {
                'num_images': len(self.psnr),
                'avg_mse': float(avg_mse),
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim),
                'std_mse': float(std_mse),
                'std_psnr': float(std_psnr),
                'std_ssim': float(std_ssim)
            },
            'per_image': per_image_results
        }
        
        json_path = os.path.join(result_dir, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存简单的文本格式结果
        txt_path = os.path.join(result_dir, "evaluation_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Number of images: {len(self.psnr)}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
            f.write(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}\n")
        
        print(f"Detailed results saved to: {json_path}")
        print(f"Summary saved to: {txt_path}")
        print(f"Images saved to: {os.path.join(cfg.result_dir, 'images')}")
        
        return {
            'avg_psnr': float(avg_psnr),
            'avg_ssim': float(avg_ssim),
            'avg_mse': float(avg_mse)
        }
