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
        检查并处理图像的数值范围
        """
        # 调试：检查图像的数值范围
        # print(f"DEBUG - img_pred range: [{img_pred.min():.6f}, {img_pred.max():.6f}]")
        # print(f"DEBUG - img_gt range: [{img_gt.min():.6f}, {img_gt.max():.6f}]")
        
        # 检查图像是否在[0,1]范围内
        pred_in_01 = (img_pred.min() >= 0) and (img_pred.max() <= 1)
        gt_in_01 = (img_gt.min() >= 0) and (img_gt.max() <= 1)
        
        if not pred_in_01 or not gt_in_01:
            print(f"WARNING: Images not in [0,1] range!")
            print(f"  Predicted image in [0,1]: {pred_in_01}")
            print(f"  Ground truth image in [0,1]: {gt_in_01}")
            
            # 如果图像在[0,255]范围，自动归一化
            if img_pred.max() > 1.0:
                print("  Auto-normalizing predicted image from [0,255] to [0,1]")
                img_pred = img_pred / 255.0
            if img_gt.max() > 1.0:
                print("  Auto-normalizing ground truth image from [0,255] to [0,1]")
                img_gt = img_gt / 255.0
        
        mse = np.mean((img_pred - img_gt) ** 2)
        # print(f"DEBUG - MSE: {mse:.8f}")
        
        if mse == 0:
            return float('inf')  # 如果MSE为0，PSNR为无穷大
        
        # 对于归一化到[0,1]的图像，MAX_I = 1
        max_pixel_value = 1.0
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
        # 由于max_pixel_value=1，所以简化为: psnr = -10 * np.log10(mse)
        
        # print(f"DEBUG - PSNR: {psnr:.2f} dB")
        
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

    def conservative_background_conversion(self, img_gt, rgb_pred):
        """
        保守的背景转换：只转换边缘的大片纯黑区域
        
        Args:
            img_gt: 真实图像 [H, W, 3]
            rgb_pred: 预测图像 [H, W, 3] (用于参考)
            
        Returns:
            转换后的GT图像
        """
        import cv2
        from scipy import ndimage
        
        H, W = img_gt.shape[:2]
        img_gt_converted = img_gt.copy()
        
        # 1. 转换到灰度图
        gray_gt = np.mean(img_gt, axis=2)
        gray_pred = np.mean(rgb_pred, axis=2)
        
        # 2. 只考虑图像边缘区域
        edge_width = min(30, H//8, W//8)  # 边缘宽度
        edge_mask = np.zeros((H, W), dtype=bool)
        edge_mask[:edge_width, :] = True     # 上边缘
        edge_mask[-edge_width:, :] = True    # 下边缘
        edge_mask[:, :edge_width] = True     # 左边缘
        edge_mask[:, -edge_width:] = True    # 右边缘
        
        # 3. 在边缘区域中寻找背景
        # 条件：非常暗 + 预测图像为白色
        very_dark_gt = gray_gt < 0.005  # 非常严格的暗区域阈值
        white_pred = gray_pred > 0.98   # 预测图像中的纯白区域
        
        # 候选背景区域
        candidate_bg = edge_mask & very_dark_gt & white_pred
        
        # 4. 连通域分析：只保留大的连通区域
        labeled, num_features = ndimage.label(candidate_bg)
        final_bg_mask = np.zeros_like(candidate_bg)
        
        if num_features > 0:
            sizes = ndimage.sum(candidate_bg, labeled, range(1, num_features + 1))
            
            # 只保留面积大于总面积2%的连通域
            min_size = H * W * 0.02
            
            for i, size in enumerate(sizes, 1):
                if size > min_size:
                    final_bg_mask[labeled == i] = True
        
        # 5. 形态学操作：平滑边界
        if final_bg_mask.sum() > 0:
            kernel = np.ones((3, 3), np.uint8)
            final_bg_mask = cv2.morphologyEx(
                final_bg_mask.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel
            ).astype(bool)
        
        # 6. 应用背景转换
        img_gt_converted[final_bg_mask] = 1.0
        
        return img_gt_converted

    def smart_background_conversion(self, img_gt, rgb_pred):
        """
        智能背景转换：更保守的背景检测，避免误伤物体黑色部分
        
        Args:
            img_gt: 真实图像 [H, W, 3]
            rgb_pred: 预测图像 [H, W, 3] (用于参考)
            
        Returns:
            转换后的GT图像
        """
        import cv2
        from scipy import ndimage
        
        H, W = img_gt.shape[:2]
        img_gt_converted = img_gt.copy()
        
        # 1. 转换到灰度图
        gray_gt = np.mean(img_gt, axis=2)
        gray_pred = np.mean(rgb_pred, axis=2)
        
        # 2. 更严格的背景检测
        # 只将非常暗且无纹理的区域视为背景
        very_dark = gray_gt < 0.01  # 更严格的阈值
        
        # 3. 预测图像中的纯白背景区域
        pred_white_bg = gray_pred > 0.95  # 预测图像中的纯白区域
        
        # 4. 边缘检测 - 检测物体边界
        edges = cv2.Canny((gray_gt * 255).astype(np.uint8), 3, 10)
        edges_dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
        near_edges = edges_dilated > 0
        
        # 5. 颜色方差分析 - 检测有纹理的区域
        # 计算每个像素邻域的方差
        kernel_size = 5
        gray_float = gray_gt.astype(np.float32)
        
        # 使用均值滤波器计算局部均值
        local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        # 计算局部方差
        local_var = cv2.blur((gray_float - local_mean) ** 2, (kernel_size, kernel_size))
        
        # 有纹理的区域（方差大于阈值）
        textured_regions = local_var > 0.001
        
        # 6. 颜色饱和度分析
        max_rgb = np.max(img_gt, axis=2)
        min_rgb = np.min(img_gt, axis=2)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        colored_regions = saturation > 0.02  # 有颜色的区域
        
        # 7. 构建更保守的前景掩码
        # 前景 = 有纹理 OR 有颜色 OR 靠近边缘 OR 不是很暗
        foreground_mask = (
            textured_regions |      # 有纹理的区域
            colored_regions |       # 有颜色的区域
            near_edges |           # 边缘附近的区域
            (gray_gt > 0.05)       # 不是很暗的区域
        )
        
        # 8. 基于预测图像的一致性检查
        # 如果预测图像在某个区域不是纯白，那么GT中对应区域很可能是前景
        pred_non_white = gray_pred < 0.9
        foreground_mask = foreground_mask | pred_non_white
        
        # 9. 连通域分析 - 只保留大的连通区域
        labeled, num_features = ndimage.label(~foreground_mask)  # 分析背景连通域
        if num_features > 0:
            sizes = ndimage.sum(~foreground_mask, labeled, range(1, num_features + 1))
            
            # 只保留面积大于总面积5%的背景连通域
            min_bg_size = H * W * 0.05
            large_bg_regions = sizes > min_bg_size
            
            # 创建新的背景掩码，只保留大的背景连通域
            refined_bg_mask = np.zeros_like(foreground_mask)
            for i, keep in enumerate(large_bg_regions, 1):
                if keep:
                    refined_bg_mask[labeled == i] = 1
            
            # 更新前景掩码
            foreground_mask = ~refined_bg_mask.astype(bool)
        
        # 10. 边界保护 - 图像边缘很可能是背景
        border_width = min(10, H//20, W//20)  # 边界宽度
        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[:border_width, :] = True     # 上边界
        border_mask[-border_width:, :] = True    # 下边界
        border_mask[:, :border_width] = True     # 左边界
        border_mask[:, -border_width:] = True    # 右边界
        
        # 边界区域如果很暗且预测图像是白色，则认为是背景
        border_background = border_mask & very_dark & pred_white_bg
        
        # 11. 最终背景掩码
        final_background_mask = border_background
        
        # 12. 形态学操作优化
        if final_background_mask.sum() > 0:
            kernel = np.ones((3, 3), np.uint8)
            final_background_mask = cv2.morphologyEx(
                final_background_mask.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel
            ).astype(bool)
        
        # 13. 应用背景转换
        img_gt_converted[final_background_mask] = 1.0
        
        # 14. 统计信息
        conversion_ratio = final_background_mask.sum() / (H * W)
        
        # 如果转换比例过高（>30%），可能有问题，进一步限制
        if conversion_ratio > 0.3:
            # 进一步限制，只转换图像边缘的大片纯黑区域
            edge_only_mask = np.zeros_like(final_background_mask)
            
            # 只在图像边缘区域寻找背景
            edge_width = min(20, H//10, W//10)
            edge_region = np.zeros((H, W), dtype=bool)
            edge_region[:edge_width, :] = True
            edge_region[-edge_width:, :] = True
            edge_region[:, :edge_width] = True
            edge_region[:, -edge_width:] = True
            
            # 在边缘区域中，只转换很暗且预测为白色的区域
            edge_background = edge_region & (gray_gt < 0.005) & (gray_pred > 0.95)
            
            # 连通域分析
            labeled, num_features = ndimage.label(edge_background)
            if num_features > 0:
                sizes = ndimage.sum(edge_background, labeled, range(1, num_features + 1))
                min_size = H * W * 0.01  # 至少1%的面积
                
                for i, size in enumerate(sizes, 1):
                    if size > min_size:
                        edge_only_mask[labeled == i] = True
            
            # 如果新的掩码更合理，使用它
            if edge_only_mask.sum() > 0 and edge_only_mask.sum() < final_background_mask.sum():
                final_background_mask = edge_only_mask
                img_gt_converted = img_gt.copy()
                img_gt_converted[final_background_mask] = 1.0
        
        return img_gt_converted

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
        
        # 调试：输出原始数据范围（在clip之前）
        # print(f"\nImage {self.img_count + 1} - Original data ranges:")
        # print(f"  rgb_pred shape: {rgb_pred.shape}, range: [{rgb_pred.min():.6f}, {rgb_pred.max():.6f}]")
        # print(f"  img_gt shape: {img_gt.shape}, range: [{img_gt.min():.6f}, {img_gt.max():.6f}]")
        # 
        # # 检查预测图像的统计信息
        # print(f"  rgb_pred mean: {rgb_pred.mean():.6f}, std: {rgb_pred.std():.6f}")
        # print(f"  img_gt mean: {img_gt.mean():.6f}, std: {img_gt.std():.6f}")
        # 
        # # 检查是否大部分像素都是0（黑色）或1（白色）
        # near_zero = (rgb_pred < 0.1).sum() / rgb_pred.size
        # near_one = (rgb_pred > 0.9).sum() / rgb_pred.size
        # print(f"  rgb_pred: {near_zero:.2%} pixels near 0, {near_one:.2%} pixels near 1")
        # 
        # gt_near_zero = (img_gt < 0.1).sum() / img_gt.size
        # gt_near_one = (img_gt > 0.9).sum() / img_gt.size
        # print(f"  img_gt: {gt_near_zero:.2%} pixels near 0, {gt_near_one:.2%} pixels near 1")
        
        # 检查是否大部分像素都是0（黑色）或1（白色）
        near_zero = (rgb_pred < 0.1).sum() / rgb_pred.size
        near_one = (rgb_pred > 0.9).sum() / rgb_pred.size
        
        gt_near_zero = (img_gt < 0.1).sum() / img_gt.size
        gt_near_one = (img_gt > 0.9).sum() / img_gt.size
        
        # 智能背景处理：精确区分物体的黑色部分和背景
        pred_bg_white = rgb_pred.mean() > 0.7 and near_one > 0.4  # 预测图像主要是白色
        gt_bg_black = img_gt.mean() < 0.4 and gt_near_zero > 0.4  # GT图像主要是黑色
        
        # 检查是否需要背景转换
        need_bg_conversion = pred_bg_white and gt_bg_black
        
        # 获取背景处理策略
        bg_strategy = getattr(cfg, 'background_strategy', 'conservative')  # 默认保守策略
        
        if need_bg_conversion:
            # 使用智能背景检测方法
            img_gt_original = img_gt.copy()
            
            if bg_strategy == 'conservative':
                # 保守策略：只转换边缘的大片纯黑区域
                img_gt = self.conservative_background_conversion(img_gt, rgb_pred)
            elif bg_strategy == 'smart':
                # 智能策略：使用复杂的特征检测
                img_gt = self.smart_background_conversion(img_gt, rgb_pred)
            elif bg_strategy == 'none':
                # 不进行背景转换
                pass
            else:
                # 默认使用保守策略
                img_gt = self.conservative_background_conversion(img_gt, rgb_pred)
            
            # 检查转换效果
            conversion_ratio = (img_gt != img_gt_original).sum() / img_gt.size
            print(f"  Applied {bg_strategy} background conversion (changed {conversion_ratio:.1%} of pixels)")
        else:
            # 如果不需要背景转换，可能只需要轻微调整
            if pred_bg_white and not gt_bg_black:
                # 预测是白背景，但GT不是黑背景，可能需要轻微调整
                pass
            elif not pred_bg_white and gt_bg_black:
                # 预测不是白背景，但GT是黑背景，可能是正常情况
                pass
        
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
    
    def render_video_if_needed(self, renderer, dataset, iteration=0):
        """
        根据配置渲染视频
        
        Args:
            renderer: volume renderer instance
            dataset: test dataset
            iteration: current iteration number
        """
        # 检查是否需要渲染视频
        if not cfg.write_video:
            return
        
        print(f"Rendering video at iteration {iteration}...")
        
        # 获取数据集的poses和相机参数
        poses = dataset.poses  # [N, 4, 4]
        
        # 获取图像尺寸和焦距
        H, W = dataset.H, dataset.W
        focal = dataset.focal
        hwf = [H, W, focal]
        
        # 构造相机内参
        intrinsics = torch.tensor([
            [focal, 0, W / 2],
            [0, focal, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # 设置输出路径
        output_dir = os.path.join(cfg.result_dir, "videos")
        exp_name = cfg.exp_name
        
        # 渲染新视角序列并生成视频
        render_type = getattr(cfg, 'render_type', 'spiral')  # 默认使用螺旋视角
        images_dir, video_path = renderer.render_novel_view_sequence(
            poses=poses,
            hwf=hwf,
            output_dir=output_dir,
            exp_name=exp_name,
            iteration=iteration,
            intrinsics=intrinsics,
            render_type=render_type
        )
        
        print(f"Novel view sequence rendered ({render_type}): {images_dir}")
        print(f"Video saved: {video_path}")
        
        # 另外，如果存在评估结果图片，也创建对比视频
        try:
            result_images_dir = os.path.join(cfg.result_dir, 'images')
            if os.path.exists(result_images_dir):
                print("Creating videos from evaluation result images...")
                
                # 创建预测图片视频
                pred_video_path = os.path.join(output_dir, f'{exp_name}_pred_sequence.mp4')
                renderer.create_video_from_result_images(
                    result_dir=cfg.result_dir,
                    output_video_path=pred_video_path,
                    image_type='pred',
                    fps=cfg.fps
                )
                
                # 创建对比视频
                comparison_video_path = os.path.join(output_dir, f'{exp_name}_comparison.mp4')
                renderer.create_video_from_result_images(
                    result_dir=cfg.result_dir,
                    output_video_path=comparison_video_path,
                    image_type='both',
                    fps=cfg.fps
                )
                
                print("Evaluation result videos created successfully!")
                
        except Exception as e:
            print(f"Error creating videos from evaluation results: {e}")
        
        print(f"Video rendering completed!")
