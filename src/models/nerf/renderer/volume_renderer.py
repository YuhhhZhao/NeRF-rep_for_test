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
        """
        Write your codes here.
        """
        print(batch.size)