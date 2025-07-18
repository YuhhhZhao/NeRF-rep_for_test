import numpy as np
import torch
from src.config import cfg


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
        
        for i in range(0, N_rays, ray_chunk_size):
            rays_o_chunk = rays_o[i:i+ray_chunk_size]
            rays_d_chunk = rays_d[i:i+ray_chunk_size]
            view_dirs_chunk = view_dirs[i:i+ray_chunk_size] if view_dirs is not None else None
            
            # 1. Coarse Sampling
            t_vals = self._sample_coarse(rays_o_chunk.shape[0])
            pts = rays_o_chunk[..., None, :] + rays_d_chunk[..., None, :] * t_vals[..., :, None]  # [N_rays_chunk, N_samples, 3]

            # 2. Query Coarse Network
            raw = self._query_network(pts, view_dirs_chunk, self.coarse_model)
            
            # 3. Volume Rendering for Coarse Pass
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
                
                # 4.3. Volume Rendering for Fine Pass
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

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape, device=self.device) * self.raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            # 调试：检查累积透明度的分布
            if torch.rand(1) < 0.01:  # 1%的概率打印调试信息
                print(f"DEBUG - acc_map stats: min={acc_map.min():.4f}, max={acc_map.max():.4f}, mean={acc_map.mean():.4f}")
                bg_contribution = (1. - acc_map).mean()
                print(f"DEBUG - background contribution mean: {bg_contribution:.4f}")
            
            # 临时禁用白色背景进行测试
            # rgb_map = rgb_map + (1. - acc_map[..., None])
            print("DEBUG - White background disabled for testing")

        return rgb_map, disp_map, acc_map, weights, depth_map
