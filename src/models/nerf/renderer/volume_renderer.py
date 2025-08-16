import numpy as np
import torch
from src.config import cfg
import imageio
import os
from tqdm import tqdm
import cv2


class Renderer:
    def __init__(self, net):
        """
        This function is responsible for defining the rendering parameters,
        including the number of samples, the step size, and the background color.
        """
        """
        Write your codes here.
        """
        self.net = net
        
        # 从 lego.yaml 的 task_arg 中读取渲染参数
        self.N_samples = cfg.task_arg.N_samples          # 64
        self.N_importance = cfg.task_arg.N_importance    # 128
        self.chunk_size = cfg.task_arg.chunk_size        # 4096
        self.white_bkgd = bool(cfg.task_arg.white_bkgd)  # True
        self.use_viewdirs = cfg.task_arg.use_viewdirs    # True
        self.lindisp = cfg.task_arg.lindisp              # False
        self.perturb = cfg.task_arg.perturb              # 1.0
        self.raw_noise_std = cfg.task_arg.raw_noise_std  # 0.0
        
        # 从 Network 获取编码器和模型
        self.embed_fn = net.embed_fn
        self.embeddirs_fn = net.embeddirs_fn
        self.coarse_model = net.model
        self.fine_model = net.model_fine
        self.device = net.device

        self.near = getattr(cfg, 'near', 2.0)
        self.far = getattr(cfg, 'far', 6.0)
        
        # ESS和ERT优化参数
        self.enable_ess = getattr(cfg, 'enable_ess', True)  # 启用空间跳过
        self.enable_ert = getattr(cfg, 'enable_ert', True)  # 启用早期射线终止
        self.ert_threshold = getattr(cfg, 'ert_threshold', 0.01)  # 早期终止阈值
        self.occupancy_grid_resolution = getattr(cfg, 'occupancy_grid_resolution', 128)  # 占用网格分辨率
        
        # 初始化占用网格（用于ESS）
        self.occupancy_grid = None
        self.grid_update_counter = 0
        self.grid_update_interval = 1000  # 每1000次调用更新一次网格

        # 初始化占用网格用于ESS
        self._initialize_occupancy_grid()


    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        The detailed rendering process is described in the paper.
        """
        
        # 从batch中获取相机参数
        H, W = int(batch['H']), int(batch['W'])
        pose = batch['pose'].squeeze(0)  # [4, 4]
        intrinsics = batch['intrinsics'].squeeze(0)  # [3, 3]
        
        # 生成像素坐标网格
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=self.device), 
                             torch.linspace(0, H-1, H, device=self.device), indexing='ij')
        i = i.t()  # [H, W]
        j = j.t()  # [H, W]
        
        # 从像素坐标计算光线方向 (相机坐标系)
        dirs = torch.stack([(i - intrinsics[0, 2]) / intrinsics[0, 0],
                           -(j - intrinsics[1, 2]) / intrinsics[1, 1],
                           -torch.ones_like(i)], -1)  # [H, W, 3]
        
        # 变换到世界坐标系
        # pose[:3, :3] 是旋转矩阵, pose[:3, 3] 是平移向量
        rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)  # [H, W, 3]
        rays_o = pose[:3, 3].expand(rays_d.shape)  # [H, W, 3]
        
        # 展平为 [H*W, 3]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        
        # 归一化光线方向
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # 视角方向 (如果需要)
        view_dirs = rays_d if self.use_viewdirs else None
        
        # 分块处理光线以避免OOM
        N_rays = rays_o.shape[0]
        ray_chunk_size = 2048  # 减小chunk size
        all_ret = {}
        
        # 初始化占用网格（首次调用时）
        if self.occupancy_grid is None:
            self._initialize_occupancy_grid()
        
        for i in range(0, N_rays, ray_chunk_size):
            rays_o_chunk = rays_o[i:i+ray_chunk_size]
            rays_d_chunk = rays_d[i:i+ray_chunk_size]
            view_dirs_chunk = view_dirs[i:i+ray_chunk_size] if view_dirs is not None else None
            
            # 1. Coarse Sampling with ESS
            if self.enable_ess:
                t_vals = self._sample_coarse_with_ess(rays_o_chunk, rays_d_chunk)
            else:
                t_vals = self._sample_coarse(rays_o_chunk.shape[0])
                
            pts = rays_o_chunk[..., None, :] + rays_d_chunk[..., None, :] * t_vals[..., :, None]

            # 2. Query Coarse Network
            raw = self._query_network(pts, view_dirs_chunk, self.coarse_model)
            
            # 3. Volume Rendering for Coarse Pass with ERT
            if self.enable_ert:
                rgb_map_0, disp_map_0, acc_map_0, weights, depth_map_0 = self._raw2outputs_with_ert(raw, t_vals, rays_d_chunk)
            else:
                rgb_map_0, disp_map_0, acc_map_0, weights, depth_map_0 = self._raw2outputs(raw, t_vals, rays_d_chunk)

            ret = {'rgb_map_0': rgb_map_0, 'disp_map_0': disp_map_0, 'acc_map_0': acc_map_0, 'depth_map_0': depth_map_0}

            # 4. Hierarchical Sampling (Fine Pass)
            if self.N_importance > 0:
                # 4.1. Importance Sampling
                t_vals_mid = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals_fine = self._sample_fine(t_vals_mid, weights[..., 1:-1])
                t_vals, _ = torch.sort(torch.cat([t_vals, t_vals_fine], -1), -1)
                pts_fine = rays_o_chunk[..., None, :] + rays_d_chunk[..., None, :] * t_vals[..., :, None]

                # 4.2. Query Fine Network
                raw_fine = self._query_network(pts_fine, view_dirs_chunk, self.fine_model)
                
                # 4.3. Volume Rendering for Fine Pass with ERT
                if self.enable_ert:
                    rgb_map, disp_map, acc_map, _, depth_map = self._raw2outputs_with_ert(raw_fine, t_vals, rays_d_chunk)
                else:
                    rgb_map, disp_map, acc_map, _, depth_map = self._raw2outputs(raw_fine, t_vals, rays_d_chunk)
                
                ret['rgb_map'] = rgb_map
                ret['disp_map'] = disp_map
                ret['acc_map'] = acc_map
                ret['depth_map'] = depth_map
            
            # 收集结果
            for k, v in ret.items():
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(v)
        
        # 拼接所有chunks的结果
        for k, v in all_ret.items():
            all_ret[k] = torch.cat(v, 0)
            # 重新reshape为图像形状
            if 'map' in k:
                if k in ['rgb_map', 'rgb_map_0']:
                    all_ret[k] = all_ret[k].view(H, W, 3)
                else:
                    all_ret[k] = all_ret[k].view(H, W)

        return all_ret

    def _sample_coarse(self, N_rays):
        """Stratified sampling along rays."""
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=self.device)
        if not self.lindisp:
            z_vals = self.near * (1. - t_vals) + self.far * (t_vals)
        else:
            z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals

    def _sample_fine(self, t_mids, weights):
        """Hierarchical sampling based on coarse weights."""
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # 检查网络的训练状态而不是renderer的训练状态
        if self.net.training:
            u = torch.rand(list(cdf.shape[:-1]) + [self.N_importance], device=self.device)
        else:
            u = torch.linspace(0., 1., steps=self.N_importance, device=self.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.N_importance])

        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(t_mids.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def _query_network(self, pts, view_dirs, model):
        """Query network in chunks to avoid OOM."""
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embedded = self.embed_fn(pts_flat)

        if self.use_viewdirs:
            view_dirs = view_dirs[:, None].expand(pts.shape)
            view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(view_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # Chunk processing
        raw = torch.cat([model(embedded[i:i+self.chunk_size]) for i in range(0, embedded.shape[0], self.chunk_size)], 0)
        raw = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
        return raw

    def _raw2outputs(self, raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values."""
        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(self.device)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # 临时修复：如果网络输出过大，进行缩放
        raw_rgb = raw[..., :3]
        # if torch.rand(1) < 0.01:  # 1%概率检查
        #     if raw_rgb.max() > 5.0:  # 如果原始输出很大
        #         print(f"DEBUG - Scaling down large raw RGB values (max: {raw_rgb.max():.2f})")
        #         raw_rgb = raw_rgb * 0.5  # 缩放因子
        
        rgb = torch.sigmoid(raw_rgb)  # [N_rays, N_samples, 3]
        
        # 调试：检查原始网络输出和sigmoid后的值
        # if torch.rand(1) < 0.01:  # 1%概率打印
        #     print(f"DEBUG - raw RGB range: [{raw[..., :3].min():.4f}, {raw[..., :3].max():.4f}]")
        #     print(f"DEBUG - sigmoid RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        #     rgb_mean = rgb.mean().item()
        #     print(f"DEBUG - sigmoid RGB mean: {rgb_mean:.4f}")
            
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=self.device) * self.raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        
        # 调试：检查密度和alpha值
        # if torch.rand(1) < 0.01:  # 1%概率打印
        #     density_raw = raw[..., 3]
        #     print(f"DEBUG - raw density range: [{density_raw.min():.4f}, {density_raw.max():.4f}]")
        #     print(f"DEBUG - alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
        #     print(f"DEBUG - alpha mean: {alpha.mean():.4f}")
        #     
        #     # 检查有多少光线有显著的alpha值
        #     significant_alpha = (alpha > 0.1).any(dim=-1).sum().item()
        #     total_rays = alpha.shape[0]
        #     print(f"DEBUG - Rays with significant alpha: {significant_alpha}/{total_rays} ({significant_alpha/total_rays:.2%})")
        
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        # 标准NeRF背景处理：根据配置文件设置添加背景颜色
        if self.white_bkgd:
            # 调试：检查累积透明度的分布
            # if torch.rand(1) < 0.02:  # 2%概率打印调试信息
            #     print(f"DEBUG - acc_map stats: min={acc_map.min():.4f}, max={acc_map.max():.4f}, mean={acc_map.mean():.4f}")
            #     incomplete_rays = (acc_map < 0.95).sum().item()
            #     total_rays = acc_map.numel()
            #     print(f"DEBUG - Incomplete rays: {incomplete_rays}/{total_rays} ({incomplete_rays/total_rays:.2%})")
            #     bg_contribution = (1. - acc_map).mean()
            #     print(f"DEBUG - Average background contribution: {bg_contribution:.4f}")
            #     
            #     # 检查rgb_map在添加背景前后的变化
            #     rgb_before = rgb_map.clone()
            #     rgb_after = rgb_map + (1. - acc_map[..., None])
            #     print(f"DEBUG - RGB before background: min={rgb_before.min():.4f}, max={rgb_before.max():.4f}, mean={rgb_before.mean():.4f}")
            #     print(f"DEBUG - RGB after background: min={rgb_after.min():.4f}, max={rgb_after.max():.4f}, mean={rgb_after.mean():.4f}")
            
            # 白色背景：添加白色到未完全不透明的区域
            rgb_map = rgb_map + (1. - acc_map[..., None])
        # 如果不使用白色背景，保持原始体积渲染结果（黑色背景）

        return rgb_map, disp_map, acc_map, weights, depth_map

    def generate_spiral_poses(self, poses, n_frames=None, n_rots=2, zrate=0.5):
        """
        Generate spiral camera path for video rendering
        
        Args:
            poses: [N, 4, 4] camera poses from dataset
            n_frames: number of frames (uses cfg.render_num if None)
            n_rots: number of rotations
            zrate: rate of up-down movement
            
        Returns:
            render_poses: [n_frames, 4, 4] spiral camera poses
        """
        if n_frames is None:
            n_frames = cfg.render_num
            
        poses = poses.cpu().numpy() if torch.is_tensor(poses) else poses
        
        # 计算相机位置的统计信息
        positions = poses[:, :3, 3]  # [N, 3]
        center = np.mean(positions, axis=0)
        
        # 计算相机的平均朝向和上向量
        forward = np.mean(poses[:, :3, 2], axis=0)
        forward = forward / np.linalg.norm(forward)
        
        up = np.mean(poses[:, :3, 1], axis=0)
        up = up / np.linalg.norm(up)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 计算相机到中心的距离
        distances = np.linalg.norm(positions - center, axis=1)
        radius = np.mean(distances)
        
        # 生成螺旋路径
        render_poses = []
        
        for i in range(n_frames):
            theta = 2 * np.pi * n_rots * i / n_frames
            phi = zrate * np.sin(2 * np.pi * i / n_frames)
            
            # 计算相机位置
            cam_pos = center + radius * (np.cos(theta) * right + np.sin(theta) * forward) + phi * up
            
            # 计算相机朝向（朝向中心）
            cam_forward = center - cam_pos
            cam_forward = cam_forward / np.linalg.norm(cam_forward)
            
            cam_right = np.cross(cam_forward, up)
            cam_right = cam_right / np.linalg.norm(cam_right)
            
            cam_up = np.cross(cam_right, cam_forward)
            
            # 构造相机变换矩阵 (OpenGL/NeRF convention)
            pose = np.eye(4)
            pose[:3, 0] = cam_right
            pose[:3, 1] = cam_up
            pose[:3, 2] = cam_forward  # 注意：NeRF使用+Z朝向，不是-Z
            pose[:3, 3] = cam_pos
            
            render_poses.append(pose)
        
        print(f"Generated {len(render_poses)} spiral poses")
        print(f"Camera positions range: {np.array([p[:3, 3] for p in render_poses]).min(axis=0)} to {np.array([p[:3, 3] for p in render_poses]).max(axis=0)}")
        print(f"Center: {center}, Radius: {radius}")
            
        return np.array(render_poses)
    
    def render_path(self, render_poses, hwf, intrinsics=None, chunk_size=None):
        """
        Render RGB and depth maps for a path of camera poses
        
        Args:
            render_poses: [N, 4, 4] camera poses
            hwf: [H, W, focal] image dimensions and focal length
            intrinsics: [3, 3] camera intrinsics (optional)
            chunk_size: batch size for rendering (optional)
            
        Returns:
            rgbs: [N, H, W, 3] rendered RGB images
            disps: [N, H, W] disparity maps
        """
        H, W, focal = hwf
        
        if intrinsics is None:
            # 构造默认相机内参
            intrinsics = torch.tensor([
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device)
        
        rgbs = []
        disps = []
        
        print(f"Rendering path with {len(render_poses)} poses...")
        
        with torch.no_grad():
            for i, pose in enumerate(tqdm(render_poses, desc="Rendering")):
                # 转换为tensor
                pose_tensor = torch.from_numpy(pose).float().to(self.device)
                intrinsics_tensor = intrinsics.to(self.device)
                
                # 构造batch
                batch = {
                    'pose': pose_tensor.unsqueeze(0),
                    'intrinsics': intrinsics_tensor.unsqueeze(0),
                    'H': H,
                    'W': W,
                }
                
                # 渲染
                try:
                    ret = self.render(batch)
                    
                    # 获取渲染结果
                    if 'rgb_map' in ret:
                        rgb_map = ret['rgb_map']
                        disp_map = ret['disp_map']
                    else:
                        rgb_map = ret['rgb_map_0']
                        disp_map = ret['disp_map_0']
                    
                    # 转换为numpy
                    rgb_np = rgb_map.cpu().numpy()
                    disp_np = disp_map.cpu().numpy()
                    
                    # 调试：检查渲染结果
                    if i % 10 == 0:  # 每10帧检查一次
                        print(f"Frame {i}: RGB range [{rgb_np.min():.4f}, {rgb_np.max():.4f}], mean={rgb_np.mean():.4f}")
                        print(f"Frame {i}: Disp range [{disp_np.min():.4f}, {disp_np.max():.4f}], mean={disp_np.mean():.4f}")
                    
                    # 确保在[0,1]范围内
                    rgb_np = np.clip(rgb_np, 0, 1)
                    disp_np = np.clip(disp_np, 0, np.max(disp_np) if np.max(disp_np) > 0 else 1.0)
                    
                    rgbs.append(rgb_np)
                    disps.append(disp_np)
                    
                except Exception as e:
                    print(f"Error rendering frame {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    # 创建黑色帧作为备用
                    rgbs.append(np.zeros((H, W, 3), dtype=np.float32))
                    disps.append(np.zeros((H, W), dtype=np.float32))
        
        return np.array(rgbs), np.array(disps)
    
    def render_novel_view_sequence(self, poses, hwf, output_dir, exp_name, iteration=0, 
                                  intrinsics=None, render_type='spiral'):
        """
        渲染新视角序列并保存为图片
        
        Args:
            poses: [N, 4, 4] camera poses from dataset
            hwf: [H, W, focal] image dimensions and focal length
            output_dir: output directory for images
            exp_name: experiment name for file naming
            iteration: current iteration number
            intrinsics: camera intrinsics (optional)
            render_type: 'spiral' or 'original' (default: 'spiral')
        """
        print(f"Rendering novel view sequence - Type: {render_type}")
        
        # 生成渲染路径
        if render_type == 'spiral':
            render_poses = self.generate_spiral_poses(poses, n_frames=cfg.render_num)
        else:
            # 使用原始poses
            render_poses = poses.cpu().numpy() if torch.is_tensor(poses) else poses
        
        # 创建输出目录
        images_dir = os.path.join(output_dir, 'novel_views')
        os.makedirs(images_dir, exist_ok=True)
        
        # 渲染并保存图片
        print(f"Rendering {len(render_poses)} novel views...")
        
        with torch.no_grad():
            for i, pose in enumerate(tqdm(render_poses, desc="Rendering novel views")):
                # 转换为tensor
                pose_tensor = torch.from_numpy(pose).float().to(self.device)
                
                if intrinsics is None:
                    H, W, focal = hwf
                    intrinsics_tensor = torch.tensor([
                        [focal, 0, W / 2],
                        [0, focal, H / 2],
                        [0, 0, 1]
                    ], dtype=torch.float32, device=self.device)
                else:
                    intrinsics_tensor = intrinsics.to(self.device)
                
                # 构造batch
                batch = {
                    'pose': pose_tensor.unsqueeze(0),
                    'intrinsics': intrinsics_tensor.unsqueeze(0),
                    'H': hwf[0],
                    'W': hwf[1],
                }
                
                # 渲染
                try:
                    ret = self.render(batch)
                    
                    # 获取渲染结果
                    if 'rgb_map' in ret:
                        rgb_map = ret['rgb_map']
                        disp_map = ret['disp_map']
                    else:
                        rgb_map = ret['rgb_map_0']
                        disp_map = ret['disp_map_0']
                    
                    # 转换为numpy并确保在[0,1]范围内
                    rgb_np = rgb_map.cpu().numpy()
                    disp_np = disp_map.cpu().numpy()
                    
                    rgb_np = np.clip(rgb_np, 0, 1)
                    disp_np = np.clip(disp_np, 0, np.max(disp_np) if np.max(disp_np) > 0 else 1.0)
                    
                    # 转换为8位格式
                    rgb_8bit = (255 * rgb_np).astype(np.uint8)
                    disp_8bit = (255 * disp_np / np.max(disp_np) if np.max(disp_np) > 0 else disp_np).astype(np.uint8)
                    
                    # 保存图片
                    rgb_path = os.path.join(images_dir, f'view{i:04d}_rgb.png')
                    disp_path = os.path.join(images_dir, f'view{i:04d}_disp.png')
                    
                    cv2.imwrite(rgb_path, rgb_8bit[..., [2, 1, 0]])  # BGR for OpenCV
                    cv2.imwrite(disp_path, disp_8bit)
                    
                    # 每10帧显示进度
                    if i % 10 == 0:
                        print(f"Rendered {i+1}/{len(render_poses)} views")
                        print(f"  RGB range: [{rgb_np.min():.4f}, {rgb_np.max():.4f}]")
                        print(f"  Saved: {rgb_path}")
                        
                except Exception as e:
                    print(f"Error rendering frame {i}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Novel view sequence saved to: {images_dir}")
        
        # 自动生成视频
        video_path = os.path.join(output_dir, f'{exp_name}_{render_type}_{iteration:06d}.mp4')
        self.create_video_from_images(
            images_dir, 
            video_path, 
            pattern='*_rgb.png',
            sort_key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', ''))
        )
        
        return images_dir, video_path
    
    def create_video_from_images(self, image_dir, output_video_path, fps=None, pattern='*.png', sort_key=None):
        """
        根据已渲染的图片创建视频
        
        Args:
            image_dir: 包含图片的目录
            output_video_path: 输出视频文件路径
            fps: 帧率 (使用cfg.fps如果为None)
            pattern: 图片文件匹配模式
            sort_key: 排序键函数 (默认按文件名排序)
        """
        import glob
        
        if fps is None:
            fps = cfg.fps
        
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
    
    def create_video_from_result_images(self, result_dir, output_video_path, image_type='pred', fps=None):
        """
        根据评估结果中的图片创建视频
        
        Args:
            result_dir: 结果目录 (通常是 cfg.result_dir)
            output_video_path: 输出视频路径
            image_type: 图片类型 ('pred', 'gt', 'both')
            fps: 帧率
        """
        images_dir = os.path.join(result_dir, 'images')
        
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return
        
        if image_type == 'pred':
            # 使用预测图片
            pattern = '*_pred.png'
            self.create_video_from_images(
                images_dir, 
                output_video_path, 
                fps=fps, 
                pattern=pattern,
                sort_key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', ''))
            )
        elif image_type == 'gt':
            # 使用真实图片
            pattern = '*_gt.png'
            gt_video_path = output_video_path.replace('.mp4', '_gt.mp4')
            self.create_video_from_images(
                images_dir, 
                gt_video_path, 
                fps=fps, 
                pattern=pattern,
                sort_key=lambda x: int(os.path.basename(x).split('_')[0].replace('view', ''))
            )
        elif image_type == 'both':
            # 创建预测和真实图片的对比视频
            self.create_comparison_video(images_dir, output_video_path, fps=fps)
    
    def create_comparison_video(self, images_dir, output_video_path, fps=None):
        """
        创建预测和真实图片的对比视频
        
        Args:
            images_dir: 图片目录
            output_video_path: 输出视频路径
            fps: 帧率
        """
        import glob
        
        if fps is None:
            fps = cfg.fps
        
        # 查找预测和真实图片
        pred_files = glob.glob(os.path.join(images_dir, '*_pred.png'))
        gt_files = glob.glob(os.path.join(images_dir, '*_gt.png'))
        
        if not pred_files or not gt_files:
            print(f"Not enough images found in {images_dir}")
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
            comparison_video_path = output_video_path.replace('.mp4', '_comparison.mp4')
            print(f"Creating comparison video: {comparison_video_path}")
            
            imageio.mimwrite(
                comparison_video_path,
                frames,
                fps=fps,
                quality=8,
                macro_block_size=16
            )
            
            print(f"Comparison video saved: {comparison_video_path}")
            
        except Exception as e:
            print(f"Error creating comparison video: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_occupancy_grid(self):
        """初始化占用网格用于ESS"""
        if not self.enable_ess:
            return
            
        print("Initializing occupancy grid for ESS...")
        
        # 创建3D占用网格
        res = self.occupancy_grid_resolution
        self.occupancy_grid = torch.ones((res, res, res), device=self.device, dtype=torch.bool)
        
        # 定义场景边界框（基于near和far）
        self.scene_bbox_min = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.scene_bbox_max = torch.tensor([1.5, 1.5, 1.5], device=self.device)
        
        print(f"Occupancy grid initialized: {res}^3 = {res**3} voxels")
        print(f"Scene bbox: {self.scene_bbox_min} to {self.scene_bbox_max}")
    
    def _update_occupancy_grid(self, pts, densities):
        """根据采样点和密度更新占用网格"""
        if not self.enable_ess or self.occupancy_grid is None:
            return
            
        # 将3D点映射到网格索引
        pts_normalized = (pts - self.scene_bbox_min) / (self.scene_bbox_max - self.scene_bbox_min)
        pts_normalized = torch.clamp(pts_normalized, 0, 1)
        
        grid_coords = (pts_normalized * (self.occupancy_grid_resolution - 1)).long()
        
        # 根据密度阈值更新占用状态
        density_threshold = 0.1
        occupied = densities > density_threshold
        
        # 更新网格
        for i in range(len(grid_coords)):
            if occupied[i]:
                x, y, z = grid_coords[i]
                self.occupancy_grid[x, y, z] = True
    
    def _is_empty_space(self, pts):
        """检查给定点是否在空间中（用于ESS）"""
        if not self.enable_ess or self.occupancy_grid is None:
            return torch.zeros(len(pts), device=self.device, dtype=torch.bool)
        
        # 将3D点映射到网格索引
        pts_normalized = (pts - self.scene_bbox_min) / (self.scene_bbox_max - self.scene_bbox_min)
        pts_normalized = torch.clamp(pts_normalized, 0, 1)
        
        grid_coords = (pts_normalized * (self.occupancy_grid_resolution - 1)).long()
        grid_coords = torch.clamp(grid_coords, 0, self.occupancy_grid_resolution - 1)
        
        # 检查占用状态
        is_empty = ~self.occupancy_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
        
        return is_empty
    
    def _sample_coarse_with_ess(self, rays_o, rays_d):
        """带ESS的粗糙采样"""
        N_rays = rays_o.shape[0]
        
        # 初始采样
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=self.device)
        if not self.lindisp:
            z_vals = self.near * (1. - t_vals) + self.far * (t_vals)
        else:
            z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * (t_vals))
        
        z_vals = z_vals.expand([N_rays, self.N_samples])
        
        if self.enable_ess and self.occupancy_grid is not None:
            # ESS: 过滤空白区域的采样点
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts_flat = pts.reshape(-1, 3)
            
            # 检查哪些点在空白区域
            is_empty = self._is_empty_space(pts_flat)
            is_empty = is_empty.reshape(N_rays, self.N_samples)
            
            # 在空白区域减少采样密度
            empty_mask = is_empty
            z_vals_filtered = []
            
            for i in range(N_rays):
                ray_z_vals = z_vals[i]
                ray_empty_mask = empty_mask[i]
                
                # 保留非空白区域的采样点
                non_empty_z_vals = ray_z_vals[~ray_empty_mask]
                
                # 如果非空白采样点太少，添加一些均匀采样点
                if len(non_empty_z_vals) < self.N_samples // 2:
                    extra_samples = self.N_samples - len(non_empty_z_vals)
                    extra_z_vals = torch.linspace(self.near, self.far, extra_samples, device=self.device)
                    combined_z_vals = torch.cat([non_empty_z_vals, extra_z_vals])
                else:
                    combined_z_vals = non_empty_z_vals
                
                # 确保采样点数量一致
                if len(combined_z_vals) > self.N_samples:
                    combined_z_vals = combined_z_vals[:self.N_samples]
                elif len(combined_z_vals) < self.N_samples:
                    # 填充到所需数量
                    padding_size = self.N_samples - len(combined_z_vals)
                    padding_z_vals = torch.linspace(self.near, self.far, padding_size, device=self.device)
                    combined_z_vals = torch.cat([combined_z_vals, padding_z_vals])
                
                z_vals_filtered.append(combined_z_vals)
            
            z_vals = torch.stack(z_vals_filtered)
        
        # 添加扰动
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals
    
    def _raw2outputs_with_ert(self, raw, z_vals, rays_d):
        """带ERT的体积渲染"""
        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(self.device)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=self.device) * self.raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)
        
        # ERT: 早期射线终止
        if self.enable_ert:
            # 计算累积透射率
            transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
            
            # 找到透射率低于阈值的位置
            terminate_mask = transmittance < self.ert_threshold
            
            # 对于应该终止的光线，将后续的alpha设为0
            for i in range(alpha.shape[0]):
                terminate_indices = torch.where(terminate_mask[i])[0]
                if len(terminate_indices) > 0:
                    first_terminate_idx = terminate_indices[0]
                    alpha[i, first_terminate_idx:] = 0.0
        
        # 标准体积渲染
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        # 背景处理
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        # 更新占用网格（用于ESS）
        if self.enable_ess and self.grid_update_counter % self.grid_update_interval == 0:
            pts = rays_d[..., None, :] * z_vals[..., :, None]  # 简化的点计算
            densities = torch.nn.functional.relu(raw[..., 3])
            self._update_occupancy_grid(pts.reshape(-1, 3), densities.reshape(-1))
        
        self.grid_update_counter += 1

        return rgb_map, disp_map, acc_map, weights, depth_map
