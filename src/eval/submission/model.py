"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn
from .train_cls import main
import os
from .PromptAD import CLIPAD

class Model(nn.Module):
    """TODO: Implement your model here"""

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        images = setup_data['few_shot_samples']
        category = setup_data['dataset_category']
        k_shot = images.shape[0]
        print("try to get seed when setup: ", torch.initial_seed())
        args = {
        "dataset": "mvtecloco",                # 数据集名称，可选['mvtec', 'visa', 'mvtecloco']
        "train_data": images,                   # 直接加载训练数据
        "class_name": category,                # 类别名称
        "img_resize": 240,                     # 图像缩放尺寸 ViT-B-16-plus-240
        "img_cropsize": 240,                   # 图像裁剪尺寸 ViT-B-16-plus-240
        "resolution": 400,                     # 分辨率
        "batch_size": 400,                     # 批量大小
        "vis": True,                           # 是否可视化（布尔值）
        "root_dir": "./src/eval/submission/result",                # 结果保存根目录
        "load_memory": True,                   # 是否加载到内存（布尔值）
        "cal_pro": False,                      # 是否计算概率（布尔值）
        # "seed": 111,                          # 随机种子
        "gpu_id": 1,                           # GPU ID
        "pure_test": False,                    # 是否仅测试模式（布尔值）
        "k_shot": k_shot,                      # Few-Shot 的样本数量
        "backbone": "ViT-B-16-plus-240",       # 骨干网络，可选['ViT-B-16-plus-240', 'ViT-B-16', ...]
        "pretrained_dataset": "laion400m_e32", # 预训练数据集名称
        "use_cpu": 0,                         # 是否使用 CPU（0=否）
        "n_ctx": 4,                           # 上下文提示词数量
        "n_ctx_ab": 1,                        # 异常检测上下文提示词数量
        "n_pro": 3,                           # 原型数量
        "n_pro_ab": 4,                        # 异常检测原型数量
        "Epoch": 50,                         # 训练轮数
        "lr": 0.002,                          # 学习率
        "momentum": 0.9,                      # 动量
        "weight_decay": 0.0005,               # 权重衰减
        "lambda1": 0.001                      # 损失函数超参数            
        }
        
        main(args)
        pass

    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        # # seed = torch.initial_seed()
        # # print("try to get seed when get urls: ", torch.initial_seed())
        # # print(seed)
        # root_dir = "./result"
        # check_dir = os.path.join(root_dir, f'{category}', 'k_1', 'checkpoint')
        # # check_path = os.path.join(check_dir, f"CLS-Seed_{seed}-{category}-check_point.pt")
        # check_path = os.path.join(check_dir, f"{'CLS'}-{category}-check_point.pt")
        model, _, _ = CLIPAD.create_model_and_transforms(model_name='ViT-B-16-plus-240', pretrained='laion400m_e32', precision = 'fp16')
        return None

    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # TODO: Implement the forward pass of the model.
        batch_size = image.shape[0]
        return ImageBatch(
            image=image,
            pred_score=torch.zeros(batch_size, device=image.device),
        )
