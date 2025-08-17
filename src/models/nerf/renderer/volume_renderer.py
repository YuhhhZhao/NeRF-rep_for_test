import numpy as np
import torch
from src.config import cfg
import imageio
import os
from tqdm import tqdm
import cv2

# 尝试导入CUDA扩展
try:
    import kilonerf_cuda
    CUDA_AVAILABLE = True
    print("CUDA extension loaded successfully!")
except ImportError as e:
    print(f"CUDA extension not available: {e}")
    CUDA_AVAILABLE = False


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
        self.ert_threshold = getattr(cfg, 'ert_threshold', 0.05)  # 增大阈值，更激进的早期终止
        self.occupancy_grid_resolution = getattr(cfg, 'occupancy_grid_resolution', 128)  # 占用网格分辨率
        
        # CUDA并行化参数
        self.use_cuda_kernels = CUDA_AVAILABLE and getattr(cfg, 'use_cuda_kernels', True)
        self.cuda_blocks = getattr(cfg, 'cuda_blocks', 128)
        self.cuda_threads = getattr(cfg, 'cuda_threads', 256)
        
        # 初始化占用网格（用于ESS）
        self.occupancy_grid = None
        self.grid_update_counter = 0
        self.grid_update_interval = 1000  # 每1000次调用更新一次网格

        # 初始化占用网格用于ESS
        self._initialize_occupancy_grid()
        
        # 初始化CUDA kernels
        if self.use_cuda_kernels:
            self._init_cuda_kernels()
            
    def _init_cuda_kernels(self):
        """初始化CUDA kernels"""
        if not CUDA_AVAILABLE:
            print("Warning: CUDA kernels requested but not available, falling back to PyTorch")
            self.use_cuda_kernels = False
            return
            
        try:
            # 初始化stream pool和magma
            kilonerf_cuda.init_stream_pool()
            kilonerf_cuda.init_magma()
            print("CUDA kernels initialized successfully")
        except Exception as e:
            print(f"Failed to initialize CUDA kernels: {e}")
            self.use_cuda_kernels = False

    def render(self, batch):
        """
        This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
        The detailed rendering process is described in the paper.
        
        优先使用CUDA并行化渲染，如果失败则回退到PyTorch实现
        """
        
        # 优先尝试CUDA并行化渲染
        if self.use_cuda_kernels:
            try:
                return self.render_cuda_parallel(batch)
            except Exception as e:
                print(f"CUDA parallel rendering failed: {e}")
                print("Falling back to PyTorch implementation...")
                self.use_cuda_kernels = False  # 禁用后续CUDA尝试
        
        # PyTorch实现 (原始方法)
        return self._render_pytorch(batch)
    
    def _render_pytorch(self, batch):
        """
        原始的PyTorch渲染实现
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
                    disp_np = np.clip(disp_np, 0, np.max(disp_np) if np.max(disp_np) > 0 else 1.0);
                    
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
        """初始化占用网格用于ESS - 优化版本"""
        if not self.enable_ess:
            return
            
        print("Initializing occupancy grid for ESS...")
        
        # 创建3D占用网格 - 初始化策略优化
        res = self.occupancy_grid_resolution
        self.occupancy_grid = torch.zeros((res, res, res), device=self.device, dtype=torch.bool)
        
        # 定义场景边界框（基于near和far，更合理的范围）
        self.scene_bbox_min = torch.tensor([-2.0, -2.0, -2.0], device=self.device)
        self.scene_bbox_max = torch.tensor([2.0, 2.0, 2.0], device=self.device)
        
        # 更智能的初始化：创建一个球形的初始占用区域
        # 这对于像Lego这样的中心化场景更合适
        grid_coords = torch.stack(torch.meshgrid([
            torch.arange(res, device=self.device),
            torch.arange(res, device=self.device), 
            torch.arange(res, device=self.device)
        ], indexing='ij'), -1).float()
        
        # 转换到标准化坐标 [-1, 1]
        grid_coords = (grid_coords / (res - 1)) * 2.0 - 1.0
        
        # 创建球形占用区域（半径0.8，覆盖大部分有效区域）
        distances = torch.norm(grid_coords, dim=-1)
        sphere_mask = distances <= 1.2  # 更大的初始区域
        
        # 同时添加一些随机噪声以避免过度规整
        random_mask = torch.rand((res, res, res), device=self.device) < 0.1  # 10%随机占用
        
        # 结合球形区域和随机区域
        self.occupancy_grid = sphere_mask | random_mask
        
        print(f"Occupancy grid initialized: {res}^3 = {res**3} voxels")
        print(f"Scene bbox: {self.scene_bbox_min} to {self.scene_bbox_max}")
        initial_occupancy = self.occupancy_grid.sum().item() / self.occupancy_grid.numel()
        print(f"Initial occupancy rate: {initial_occupancy:.4f} ({initial_occupancy*100:.2f}%)")
        
        # 设置自适应参数
        self.ess_skip_threshold = 0.5  # 降低跳过阈值，更激进地使用ESS
        self.grid_update_interval = 500  # 降低更新频率
    
    def _populate_occupancy_grid_kilonerf_method(self):
        """
        KiloNeRF方法：对每个网格单元，在3x3x3子网格上采样密度
        如果任何密度超过阈值τ则标记为占用
        """
        if not self.enable_ess or self.coarse_model is None:
            return
            
        print("Populating occupancy grid using KiloNeRF 3x3x3 subgrid sampling...")
        
        res = self.occupancy_grid_resolution
        density_threshold = 0.01  # 阈值τ，论文中说无需数据集特定调整
        
        # 计算网格单元大小
        bbox_size = self.scene_bbox_max - self.scene_bbox_min
        cell_size = bbox_size / res
        
        # 重置占用网格
        self.occupancy_grid.fill_(False)
        
        with torch.no_grad():
            # 分批处理以避免内存问题
            batch_size = 512  # 每次处理的网格单元数
            
            for batch_start in range(0, res**3, batch_size):
                batch_end = min(batch_start + batch_size, res**3)
                
                # 计算当前批次的网格索引
                batch_indices = []
                batch_points = []
                
                for flat_idx in range(batch_start, batch_end):
                    # 将平坦索引转换为3D索引
                    z = flat_idx // (res * res)
                    y = (flat_idx % (res * res)) // res
                    x = flat_idx % res
                    
                    # 计算网格单元的边界
                    cell_min = self.scene_bbox_min + torch.tensor([x, y, z], device=self.device) * cell_size
                    
                    # 在单元格内生成3x3x3采样点
                    for dz in range(3):
                        for dy in range(3):
                            for dx in range(3):
                                # 子网格内的相对位置 [0, 0.5, 1]
                                offset = torch.tensor([dx, dy, dz], device=self.device, dtype=torch.float32) / 2.0
                                point = cell_min + offset * cell_size
                                batch_points.append(point)
                                batch_indices.append((x, y, z))
                
                if not batch_points:
                    continue
                
                # 批量查询网络
                pts_tensor = torch.stack(batch_points)  # [N_points, 3]
                embedded = self.embed_fn(pts_tensor)
                
                # 分块查询模型
                densities = []
                for i in range(0, embedded.shape[0], self.chunk_size):
                    chunk = embedded[i:i+self.chunk_size]
                    raw_output = self.coarse_model(chunk)
                    density = torch.nn.functional.relu(raw_output[..., 3])  # 密度通道，确保非负
                    densities.append(density)
                
                densities = torch.cat(densities, 0)
                
                # 按网格单元分组（每个单元27个点）
                densities_by_cell = densities.view(-1, 27)  # [N_cells, 27]
                
                # 检查每个单元是否有任何密度超过阈值
                max_densities_per_cell = densities_by_cell.max(dim=1)[0]
                occupied_mask = max_densities_per_cell > density_threshold
                
                # 更新占用网格
                unique_indices = list(set(batch_indices))
                for i, (x, y, z) in enumerate(unique_indices):
                    if i < len(occupied_mask) and occupied_mask[i]:
                        self.occupancy_grid[x, y, z] = True
                
                # 显示进度
                if batch_start % (batch_size * 10) == 0:
                    progress = batch_start / (res**3) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_start}/{res**3} cells)")
        
        final_occupancy = self.occupancy_grid.sum().item() / self.occupancy_grid.numel()
        print(f"KiloNeRF occupancy grid completed. Occupancy rate: {final_occupancy:.4f} ({final_occupancy*100:.2f}%)")
    
    def _update_occupancy_grid(self, pts, densities):
        """根据采样点和密度更新占用网格 - 高效批量版本"""
        if not self.enable_ess or self.occupancy_grid is None or len(pts) == 0:
            return
            
        # 将3D点映射到网格索引
        pts_normalized = (pts - self.scene_bbox_min) / (self.scene_bbox_max - self.scene_bbox_min)
        pts_normalized = torch.clamp(pts_normalized, 0, 1)
        
        grid_coords = (pts_normalized * (self.occupancy_grid_resolution - 1)).long()
        grid_coords = torch.clamp(grid_coords, 0, self.occupancy_grid_resolution - 1)
        
        # 根据密度阈值确定占用状态
        density_threshold = 0.01  # 降低阈值以捕获更多细节
        occupied = densities > density_threshold
        
        # 批量更新网格 - 避免循环以提高效率
        if occupied.any():
            occupied_coords = grid_coords[occupied]
            # 使用高级索引批量更新
            self.occupancy_grid[occupied_coords[:, 0], occupied_coords[:, 1], occupied_coords[:, 2]] = True
            
        # 可选：逐渐衰减占用状态，允许网格自我优化
        # 这可以帮助去除不再有效的占用区域
        if hasattr(self, 'grid_decay_rate'):
            decay_rate = getattr(self, 'grid_decay_rate', 0.99)  # 每次更新保留99%
            if torch.rand(1) < 0.1:  # 10%概率进行衰减
                self.occupancy_grid = self.occupancy_grid & (torch.rand_like(self.occupancy_grid.float()) < decay_rate)
    
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
        """带ESS的粗糙采样 - 高性能版本"""
        N_rays = rays_o.shape[0]
        
        # 初始采样
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=self.device)
        if not self.lindisp:
            z_vals = self.near * (1. - t_vals) + self.far * (t_vals)
        else:
            z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * (t_vals))
        
        z_vals = z_vals.expand([N_rays, self.N_samples])
        
        if self.enable_ess and self.occupancy_grid is not None:
            # ESS: 高效的批量空白区域检测
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
            pts_flat = pts.reshape(-1, 3)
            
            # 批量检查空白区域
            is_empty = self._is_empty_space(pts_flat).reshape(N_rays, self.N_samples)
            
            # 只对真正大量空白的光线进行优化
            empty_ratios = is_empty.float().mean(dim=1)
            skip_threshold = getattr(self, 'ess_skip_threshold', 0.5)
            highly_empty_rays = empty_ratios > skip_threshold
            
            # 对于需要优化的光线，智能地重新分布采样点
            if highly_empty_rays.any():
                for i in range(N_rays):
                    if highly_empty_rays[i]:
                        ray_is_empty = is_empty[i]
                        ray_z_vals = z_vals[i]
                        
                        # 保留非空区域的采样点
                        occupied_z_vals = ray_z_vals[~ray_is_empty]
                        
                        if len(occupied_z_vals) > 0:
                            # 在非空区域周围密集采样
                            min_occupied = occupied_z_vals.min()
                            max_occupied = occupied_z_vals.max()
                            
                            # 保留原有的非空采样点
                            n_keep = len(occupied_z_vals)
                            # 在有效区域内补充采样点
                            n_add = max(0, self.N_samples - n_keep)
                            
                            if n_add > 0:
                                # 在有效区域内均匀采样
                                additional_z_vals = torch.linspace(
                                    min_occupied, max_occupied, n_add, device=self.device
                                )
                                combined_z_vals = torch.cat([occupied_z_vals, additional_z_vals])
                            else:
                                combined_z_vals = occupied_z_vals
                            
                            # 确保采样点数量正确
                            if len(combined_z_vals) > self.N_samples:
                                # 随机采样到指定数量
                                indices = torch.randperm(len(combined_z_vals))[:self.N_samples]
                                combined_z_vals = combined_z_vals[indices]
                            elif len(combined_z_vals) < self.N_samples:
                                # 如果还是不够，用原始分布补齐
                                n_pad = self.N_samples - len(combined_z_vals)
                                pad_z_vals = ray_z_vals[:n_pad]
                                combined_z_vals = torch.cat([combined_z_vals, pad_z_vals])
                            
                            # 排序并更新
                            combined_z_vals, _ = torch.sort(combined_z_vals)
                            z_vals[i] = combined_z_vals
        
        # 添加扰动
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals
    
    def _raw2outputs_with_ert(self, raw, z_vals, rays_d):
        """带ERT的体积渲染 - 极致优化版本，减少不必要计算"""
        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(self.device)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=self.device) * self.raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)
        
        # 超高效ERT：只在真正必要时计算，减少不必要开销
        N_rays, N_samples = raw.shape[:2]
        
        # 标准权重计算
        alpha_shifted = torch.cat([torch.zeros((N_rays, 1), device=self.device), alpha[:, :-1]], dim=1)
        transmittance = torch.cumprod(1.0 - alpha_shifted, dim=1)
        weights = alpha * transmittance
        
        # 简化的ERT：只在transmittance很低时将后续权重置零
        # 这样避免了复杂的掩码操作
        low_transmittance = transmittance < self.ert_threshold
        if low_transmittance.any():
            # 找到每条光线第一个低透射率的位置
            first_termination = low_transmittance.float().argmax(dim=1)
            # 创建终止掩码
            sample_indices = torch.arange(N_samples, device=self.device).expand(N_rays, -1)
            terminate_mask = sample_indices >= first_termination.unsqueeze(1)
            # 将终止位置之后的权重置零（但保留已计算的权重，避免重计算）
            weights = weights * (~terminate_mask).float()
        
        # 计算最终渲染结果
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        
        # 背景处理
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        # 极简统计（每500次显示一次，减少I/O开销）
        if hasattr(self, 'ert_debug_counter'):
            self.ert_debug_counter += 1
        else:
            self.ert_debug_counter = 1
            
        if self.ert_debug_counter % 500 == 0:  # 大幅减少统计频率
            if low_transmittance.any():
                termination_rate = low_transmittance.any(dim=1).float().mean().item() * 100
                print(f"  ERT: {termination_rate:.1f}% rays terminated early")

        # 更新占用网格（用于ESS）- 降低更新频率以提高性能
        if self.enable_ess and self.grid_update_counter % self.grid_update_interval == 0:
            # 只使用有效的采样点来更新网格
            effective_mask = weights > 1e-4  # 只考虑有显著权重的点
            if effective_mask.any():
                effective_pts = (rays_d[..., None, :] * z_vals[..., :, None])[effective_mask]
                effective_densities = torch.nn.functional.relu(raw[..., 3])[effective_mask]
                self._update_occupancy_grid(effective_pts, effective_densities)
        
        self.grid_update_counter += 1

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_cuda_parallel(self, batch):
        """
        使用CUDA kernels的并行渲染方法
        每个GPU核心负责一条光线的渲染计算
        """
        if not self.use_cuda_kernels:
            # 回退到标准实现
            return self.render(batch)
            
        # 从batch中获取相机参数
        H, W = int(batch['H']), int(batch['W'])
        pose = batch['pose'].squeeze(0)  # [4, 4]
        intrinsics = batch['intrinsics'].squeeze(0)  # [3, 3]
        
        # 提取相机参数
        fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
        cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
        
        # 相机到世界的变换矩阵 (只取旋转部分)
        c2w = pose[:3, :3].contiguous()  # [3, 3]
        origin = pose[:3, 3].contiguous()  # [3]
        
        try:
            # Step 1: 使用CUDA kernel生成光线方向
            rays_d = kilonerf_cuda.get_rays_d(
                H, W, cx, cy, fx, fy, c2w, 
                self.cuda_blocks, self.cuda_threads
            )  # [H, W, 3]
            
            # Step 2: 生成查询索引 (基于深度采样)
            total_samples = H * W * self.N_samples
            query_indices = self._generate_query_indices_cuda(H, W, self.N_samples)
            
            # Step 3: 使用CUDA kernel计算Fourier特征
            positions, directions = self._generate_positions_directions_cuda(
                query_indices, rays_d, origin, H, W
            )
            
            # Step 4: 计算位置和方向的Fourier embedding
            pos_embeddings = self._compute_fourier_features_cuda(positions, is_position=True)
            dir_embeddings = self._compute_fourier_features_cuda(directions, is_position=False)
            
            # Step 5: 网络评估 (使用CUDA优化的网络前向传播)
            rgb_sigma = self._network_eval_cuda(pos_embeddings, dir_embeddings, query_indices)
            
            # Step 6: 体积积分 (使用CUDA kernel进行并行积分)
            rgb_map, acc_map = self._integrate_cuda(rgb_sigma, rays_d, H, W)
            
            # Step 7: 背景处理
            if self.white_bkgd:
                background_color = torch.ones(3, device=self.device)
                kilonerf_cuda.replace_transparency_by_background_color(
                    rgb_map.data_ptr(), acc_map, background_color,
                    self.cuda_blocks, self.cuda_threads
                )
            
            # 重新调整为图像形状
            rgb_map = rgb_map.view(H, W, 3)
            acc_map = acc_map.view(H, W)
            
            # 计算深度和视差图 (简化版本)
            depth_map = torch.zeros(H, W, device=self.device)  # TODO: 从积分中计算
            disp_map = 1. / torch.clamp(depth_map, min=1e-10)
            
            return {
                'rgb_map': rgb_map,
                'depth_map': depth_map,
                'disp_map': disp_map,
                'acc_map': acc_map
            }
            
        except Exception as e:
            print(f"CUDA rendering failed: {e}, falling back to PyTorch implementation")
            return self.render(batch)
    
    def _generate_query_indices_cuda(self, H, W, N_samples):
        """生成查询索引用于CUDA kernel"""
        total_queries = H * W * N_samples
        
        # 创建查询索引: (y * W + x) * N_samples + depth_idx
        query_indices = torch.zeros(total_queries, dtype=torch.int32, device=self.device)
        
        idx = 0
        for y in range(H):
            for x in range(W):
                for d in range(N_samples):
                    query_indices[idx] = (y * W + x) * N_samples + d
                    idx += 1
                    
        return query_indices
    
    def _generate_positions_directions_cuda(self, query_indices, rays_d, origin, H, W):
        """根据查询索引生成3D位置和方向"""
        N_queries = query_indices.shape[0]
        
        # 解码查询索引获取像素坐标和深度索引
        positions = torch.zeros(N_queries, 3, device=self.device)
        directions = torch.zeros(N_queries, 3, device=self.device)
        
        # 深度采样
        t_vals = torch.linspace(0., 1., steps=self.N_samples, device=self.device)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals
        
        for i, query_idx in enumerate(query_indices):
            # 解码查询索引
            depth_idx = query_idx % self.N_samples
            pixel_idx = query_idx // self.N_samples
            y = pixel_idx // W
            x = pixel_idx % W
            
            # 获取光线方向和深度
            ray_d = rays_d[y, x]  # [3]
            z_val = z_vals[depth_idx]
            
            # 计算3D位置
            positions[i] = origin + z_val * ray_d
            directions[i] = ray_d / torch.norm(ray_d)  # 标准化方向
            
        return positions, directions
    
    def _compute_fourier_features_cuda(self, input_tensor, is_position=True):
        """使用CUDA kernel计算Fourier特征"""
        if is_position:
            # 位置编码: 10个频率
            num_frequencies = 10
            frequency_bands = torch.tensor([2.**i for i in range(num_frequencies)], 
                                         device=self.device, dtype=torch.float32)
        else:
            # 方向编码: 4个频率
            num_frequencies = 4  
            frequency_bands = torch.tensor([2.**i for i in range(num_frequencies)], 
                                         device=self.device, dtype=torch.float32)
        
        # 为每个输入维度计算Fourier特征
        input_flat = input_tensor.flatten()  # 展平所有输入
        
        try:
            embeddings = kilonerf_cuda.compute_fourier_features(
                input_flat, frequency_bands, 
                self.cuda_blocks, self.cuda_threads, "default"
            )
            
            # 重新调整形状: [N_points, input_dim * (2*num_freq + 1)]
            input_dim = input_tensor.shape[-1]
            embed_dim = input_dim * (2 * num_frequencies + 1)
            embeddings = embeddings.view(-1, embed_dim)
            
            return embeddings
            
        except Exception as e:
            print(f"CUDA Fourier features failed: {e}, using PyTorch fallback")
            # PyTorch fallback
            return self._compute_fourier_features_pytorch(input_tensor, num_frequencies)
    
    def _compute_fourier_features_pytorch(self, input_tensor, num_frequencies):
        """PyTorch实现的Fourier特征 (fallback)"""
        frequency_bands = torch.tensor([2.**i for i in range(num_frequencies)], 
                                     device=self.device, dtype=torch.float32)
        
        embeddings = [input_tensor]
        for freq in frequency_bands:
            embeddings.append(torch.cos(freq * input_tensor))
            embeddings.append(torch.sin(freq * input_tensor))
            
        return torch.cat(embeddings, dim=-1)
    
    def _network_eval_cuda(self, pos_embeddings, dir_embeddings, query_indices):
        """使用CUDA优化的网络评估"""
        # TODO: 实现CUDA网络评估
        # 目前使用PyTorch fallback
        return self._network_eval_pytorch(pos_embeddings, dir_embeddings)
    
    def _network_eval_pytorch(self, pos_embeddings, dir_embeddings):
        """PyTorch网络评估 (fallback)"""
        N_points = pos_embeddings.shape[0]
        
        # 组合位置和方向embeddings
        if self.use_viewdirs:
            full_embeddings = torch.cat([pos_embeddings, dir_embeddings], dim=-1)
        else:
            full_embeddings = pos_embeddings
            
        # 分块处理避免OOM
        rgb_sigma_list = []
        for i in range(0, N_points, self.chunk_size):
            chunk = full_embeddings[i:i+self.chunk_size]
            rgb_sigma_chunk = self.coarse_model(chunk)
            rgb_sigma_list.append(rgb_sigma_chunk)
            
        rgb_sigma = torch.cat(rgb_sigma_list, dim=0)
        return rgb_sigma  # [N_points, 4]
    
    def _integrate_cuda(self, rgb_sigma, rays_d, H, W):
        """使用CUDA kernel进行体积积分"""
        N_rays = H * W
        N_samples = self.N_samples
        
        # 准备数据
        rgb_sigma_tensor = rgb_sigma.view(N_rays, N_samples, 4)  # [N_rays, N_samples, 4]
        rgb_sigma_flat = rgb_sigma_tensor.contiguous()
        
        # 计算光线间距
        t_vals = torch.linspace(0., 1., steps=N_samples, device=self.device)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals
        dists = z_vals[1:] - z_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=self.device)])
        dists = dists.expand(N_rays, -1).contiguous()
        
        # 输出缓冲区
        rgb_map = torch.zeros(N_rays, 3, device=self.device)
        acc_map = torch.zeros(N_rays, device=self.device)
        transmittance = torch.ones(N_rays, device=self.device)
        active_ray_mask = torch.ones(N_rays, dtype=torch.bool, device=self.device)
        
        try:
            # 调用CUDA积分kernel
            kilonerf_cuda.integrate(
                rgb_sigma_flat, dists, rgb_map.data_ptr(),
                acc_map, transmittance, active_ray_mask,
                N_rays, N_samples, self.ert_threshold, True,
                self.cuda_blocks, self.cuda_threads, 0  # version 0
            )
            
            return rgb_map, acc_map
            
        except Exception as e:
            print(f"CUDA integration failed: {e}, using PyTorch fallback")
            return self._integrate_pytorch(rgb_sigma_tensor, z_vals)
    
    def _integrate_pytorch(self, rgb_sigma, z_vals):
        """PyTorch体积积分实现 (fallback)"""
        N_rays, N_samples = rgb_sigma.shape[:2]
        
        # 计算距离
        dists = z_vals[1:] - z_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=self.device)])
        dists = dists.expand(N_rays, -1)
        
        # 提取RGB和density
        rgb = torch.sigmoid(rgb_sigma[..., :3])
        sigma = torch.nn.functional.relu(rgb_sigma[..., 3])
        
        # 计算alpha
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        # 计算权重
        alpha_shifted = torch.cat([torch.zeros((N_rays, 1), device=self.device), alpha[:, :-1]], dim=1)
        transmittance = torch.cumprod(1.0 - alpha_shifted, dim=1)
        weights = alpha * transmittance
        
        # 体积渲染
        rgb_map = torch.sum(weights[..., None] * rgb, dim=1)  # [N_rays, 3]
        acc_map = torch.sum(weights, dim=1)  # [N_rays]
        
        return rgb_map, acc_map

    def __del__(self):
        """清理CUDA资源"""
        if self.use_cuda_kernels and CUDA_AVAILABLE:
            try:
                kilonerf_cuda.destroy_stream_pool()
            except:
                pass  # 忽略清理错误
