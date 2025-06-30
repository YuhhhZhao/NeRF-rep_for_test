import torch.utils.data as data
import torch
import numpy as np
from src.config import cfg
import os
import json
import imageio

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        """
        Write your codes here.
        """
        #import ipdb; ipdb.set_trace()  # 调试用
        self.data_root = kwargs.get("data_root", "data/nerf_synthetic")
        self.split = kwargs.get("split", "train")
        self.scene = kwargs.get("scene", "lego")
        self.input_ratio = kwargs.get("input_ratio", 1.0)
        self.cams= kwargs.get("cams", None)
        self.H = kwargs.get("H", 800)
        self.W = kwargs.get("W", 800)

        self.scene_path = os.path.join(self.data_root, self.scene)
    
        transforms_file = f'transforms_{self.split}.json'
        transforms_path = os.path.join(self.scene_path, transforms_file)
        
        with open(transforms_path, 'r') as f:
            self.meta = json.load(f)
        # 计算相机内参
        if 'camera_angle_x' in self.meta:
            self.focal = 0.5 * self.W / np.tan(0.5 * self.meta['camera_angle_x'])
        else:
            self.focal = 800  
        
        # 加载所有图像和相机位姿
        self.images = []
        self.poses = []
        
        for frame in self.meta['frames']:
            # 加载图像
            img_path = os.path.join(self.scene_path, frame['file_path'] + '.png')
            if os.path.exists(img_path):
                img = imageio.imread(img_path)
                import ipdb; ipdb.set_trace()  # 调试用
                # 转换为torch tensor
                img = torch.from_numpy(img).float()
                
                # 处理 RGBA -> RGB
                if img.shape[-1] == 4:
                    # 使用 alpha 混合
                    alpha = img[..., 3:4] / 255.0
                    img = img[..., :3] * alpha + (1 - alpha) * 255
                
                img = img / 255.0
                if img.shape[:2] != (self.H, self.W):
                    # 需要调整维度顺序: [H, W, C] -> [C, H, W] -> [1, C, H, W]
                    img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                    img = torch.nn.functional.interpolate(
                        img, size=(self.H, self.W), mode='bilinear', align_corners=False
                    )
                    img = img.squeeze(0).permute(1, 2, 0)  # [H, W, C]
                
                self.images.append(img)
                
                # 加载相机位姿
                pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
                self.poses.append(pose)
        
        # 转换为 tensor stack
        if self.images:
            self.images = torch.stack(self.images)  # [N, H, W, 3]
            self.poses = torch.stack(self.poses)    # [N, 4, 4]
        else:
            self.images = torch.empty(0, self.H, self.W, 3)
            self.poses = torch.empty(0, 4, 4)
        

    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典
        """
        """
        Write your codes here.
        """
        # 获取对应的图像和位姿
        image = self.images[index]  # [H, W, 3]
        pose = self.poses[index]    # [4, 4]
        
        # 构造相机内参矩阵
        intrinsics = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        ret = {
            "index": index,
            "image": image,           # 目标图像 [H, W, 3]
            "pose": pose,             # 相机外参 [4, 4]
            "intrinsics": intrinsics, # 相机内参 [3, 3]
            "H": self.H,
            "W": self.W,
            "focal": self.focal,
            "scene": self.scene,
            "split": self.split,
        }
        return ret

    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        """
        Write your codes here.
        """
        return len(self.images) if hasattr(self, 'images') and len(self.images) > 0 else 0
