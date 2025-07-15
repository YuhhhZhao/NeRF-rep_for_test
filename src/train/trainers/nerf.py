import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.nerf.renderer.volume_renderer import Renderer


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)
        self.train_loader = train_loader
        self.device = next(self.net.parameters()).device
        
        # 添加训练指标
        self.train_loss = 0.0
        self.train_psnr = 0.0
        self.train_steps = 0

    def forward(self, batch):
        """
        Forward pass for training and inference.
        """
        # 将batch数据移动到GPU
        for k, v in batch.items():
            if k != 'meta' and torch.is_tensor(v):
                batch[k] = v.to(self.device)
        
        # 渲染输出
        output = self.renderer.render(batch)
        
        # 如果在训练模式且有真实图像，计算损失
        if self.training and 'image' in batch:
            loss_dict = self.compute_loss(output, batch)
            output.update(loss_dict)
        
        return output

    def compute_loss(self, output, batch):
        """
        计算NeRF的损失函数
        """
        # 获取真实图像
        target_rgb = batch['image'].squeeze(0)  # [H, W, 3] or [H, W, 4]
        if target_rgb.shape[-1] == 4:  # 如果有alpha通道，只取RGB
            target_rgb = target_rgb[..., :3]
        
        # 展平为[H*W, 3]
        target_rgb = target_rgb.view(-1, 3)
        
        # 粗糙网络损失
        rgb_coarse = output['rgb_map_0']
        loss_coarse = F.mse_loss(rgb_coarse, target_rgb)
        
        loss_dict = {
            'loss_coarse': loss_coarse,
            'loss': loss_coarse
        }
        
        # 精细网络损失（如果存在）
        if 'rgb_map' in output:
            rgb_fine = output['rgb_map']
            loss_fine = F.mse_loss(rgb_fine, target_rgb)
            loss_dict['loss_fine'] = loss_fine
            loss_dict['loss'] = loss_coarse + loss_fine
        
        # 计算PSNR
        with torch.no_grad():
            if 'rgb_map' in output:
                mse = F.mse_loss(output['rgb_map'], target_rgb)
            else:
                mse = F.mse_loss(output['rgb_map_0'], target_rgb)
            psnr = -10. * torch.log10(mse)
            loss_dict['psnr'] = psnr
        
        return loss_dict

    def update_metrics(self, loss_dict):
        """
        更新训练指标
        """
        self.train_steps += 1
        self.train_loss += loss_dict['loss'].item()
        if 'psnr' in loss_dict:
            self.train_psnr += loss_dict['psnr'].item()

    def get_metrics(self):
        """
        获取平均训练指标
        """
        if self.train_steps == 0:
            return {'avg_loss': 0.0, 'avg_psnr': 0.0}
        
        return {
            'avg_loss': self.train_loss / self.train_steps,
            'avg_psnr': self.train_psnr / self.train_steps
        }

    def reset_metrics(self):
        """
        重置训练指标
        """
        self.train_loss = 0.0
        self.train_psnr = 0.0
        self.train_steps = 0

    def train_step(self, batch):
        """
        单步训练
        """
        self.train()
        output = self.forward(batch)
        
        if 'loss' in output:
            self.update_metrics(output)
            return output['loss']
        else:
            # 如果forward没有计算损失，在这里计算
            loss_dict = self.compute_loss(output, batch)
            self.update_metrics(loss_dict)
            return loss_dict['loss']

    def eval_step(self, batch):
        """
        单步评估
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
            if 'image' in batch:
                loss_dict = self.compute_loss(output, batch)
                output.update(loss_dict)
            return output
