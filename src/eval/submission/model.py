"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn
from .train_cls import main
import os
from .PromptAD import CLIPAD
import numpy as np

import torch
import random
import torch.nn as nn
from .PromptAD import CLIPAD
from torch.nn import functional as F
from .PromptAD.ad_prompts import *
from PIL import Image
from scipy.ndimage import gaussian_filter

from .PromptAD.CLIPAD import SimpleTokenizer as _Tokenizer
from .PromptAD import *
from .utils.training_utils import *

_tokenizer = _Tokenizer()   # local tokenizer, no padding, no sos, no eos

# valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336", 'ViT-B-32-plus-256']
valid_pretrained_datasets = ['laion400m_e32', 'openai']

from torchvision import transforms


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert('RGB')


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre):
        super().__init__()

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(classname)
        state_anomaly1 = state_anomaly + class_state_abnormal[classname]

        if classname in class_mapping:
            classname = class_mapping[classname]

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_handle = len(state_anomaly1)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_handle

        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_handle = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_handle])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = clip_model.token_embedding(tokenized_abnormal_prompts_handle).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_handle, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_handle
        # self.name_lens = name_lens

    def forward(self):

        # learned normal prompt
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # handle abnormal prompt
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_handle = self.abnormal_token_prefix_handle
        abnormal_suffix_handle = self.abnormal_token_suffix_handle

        abnormal_prompts_handle = torch.cat(
            [
                abnormal_prefix_handle,     # (n_pro * n_ab_handle, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_handle, n_ctx, dim)
                abnormal_suffix_handle,     # (n_pro * n_ab_handle, *, dim)
            ],
            dim=1,
        )

        # learned abnormal prompt
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_handle, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_handle

        return normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned


# class PromptAD(torch.nn.Module):

class PromptAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(PromptAD, self).__init__()

        self.shot = kwargs['k_shot']

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
        self.phrase_form = '{}'
        self.device = device

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx=4, n_pro=3, n_ctx_ab=1, n_pro_ab=4, classname=class_name, clip_model=model, pre='fp16') #self.precision)
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        self.feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1))

        b2, n2, d2 = features2.shape
        self.feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0]
            global_normality_and_abnormality_score = (t * global_feature @ self.text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[1].shape[0]

        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def forward(self, images, task):

        visual_features = self.encode_image(images)
        if task == 'seg':
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            #
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
            # anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            # anomaly_map = visual_anomaly_map
            # anomaly_map = textual_anomaly_map

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            am_img_list = []
            for i in range(textual_anomaly.shape[0]):
                am_img_list.append(textual_anomaly[i])

            return am_img_list, am_pix_list
        else:
            assert 'task error'

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()




class Model(nn.Module):
    """TODO: Implement your model here"""

    def __init__(self):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        # super(Model, self).__init__()

        # self.out_size_h = None
        # self.out_size_w = None
        # self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        # # self.device = device
        # # self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
        # self.phrase_form = '{}'
        # # self.device = device

        # # version v1: no norm for each of linguistic embedding
        # # version v1:    norm for each of linguistic embedding
        # self.version = 'V1' # V1:
        # # visual textual, textual_visual
        # device = f"cuda:0"
        # self.device = device
        # self.shot = None
        # self.transform = None
        # self.gt_transform = None

        # # self.transform = transforms.Compose([
        # #     transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
        # #     transforms.CenterCrop(kwargs['img_cropsize']),
        # #     _convert_to_rgb,
        # #     transforms.ToTensor(),
        # #     transforms.Normalize(mean=mean_train, std=std_train)])

        # # self.gt_transform = transforms.Compose([
        # #     transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
        # #     transforms.CenterCrop(kwargs['img_cropsize']),
        # #     transforms.ToTensor()])
        super().__init__()
        device = f"cuda:0"
        self.device = torch.device(device)
        self._is_trained = False  # 训练状态标志
        self.optimizer = None      # 优化器
        self.criterion = None      # 损失函数
        self.train_loader = None   # 数据加载器
        self.epochs = 0           # 训练轮数
        self.prompt_ad = None      # 核心模型

    # def __init__(self):
    #     super(Model, self).__init__()
    #     self.category = None
    #     self.k_shot = None

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        images = setup_data['few_shot_samples']
        category = setup_data['dataset_category']
        k_shot = images.shape[0]
        device = f"cuda:0"

        # """集成训练配置与初始化"""
        # # 清除已有模型
        # if self.prompt_ad is not None:
        #     del self.prompt_ad
        # torch.cuda.empty_cache()
        # args = {
        # "dataset": "mvtecloco",                # 数据集名称，可选['mvtec', 'visa', 'mvtecloco']
        # "train_data": images,                   # 直接加载训练数据
        # "class_name": category,                # 类别名称
        # "img_resize": 240,                     # 图像缩放尺寸 ViT-B-16-plus-240
        # "img_cropsize": 240,                   # 图像裁剪尺寸 ViT-B-16-plus-240
        # "resolution": 400,                     # 分辨率
        # "batch_size": 400,                     # 批量大小
        # "vis": True,                           # 是否可视化（布尔值）
        # "root_dir": "./src/eval/submission/result",                # 结果保存根目录
        # "load_memory": True,                   # 是否加载到内存（布尔值）
        # "cal_pro": False,                      # 是否计算概率（布尔值）
        # # "seed": 111,                          # 随机种子
        # "gpu_id": 1,                           # GPU ID
        # "pure_test": False,                    # 是否仅测试模式（布尔值）
        # "k_shot": k_shot,                      # Few-Shot 的样本数量
        # "backbone": "ViT-B-16-plus-240",       # 骨干网络，可选['ViT-B-16-plus-240', 'ViT-B-16', ...]
        # "pretrained_dataset": "laion400m_e32", # 预训练数据集名称
        # "use_cpu": 0,                         # 是否使用 CPU（0=否）
        # "n_ctx": 4,                           # 上下文提示词数量
        # "n_ctx_ab": 1,                        # 异常检测上下文提示词数量
        # "n_pro": 3,                           # 原型数量
        # "n_pro_ab": 4,                        # 异常检测原型数量
        # "Epoch": 3,                         # 训练轮数
        # "lr": 0.02,                          # 学习率
        # "momentum": 0.9,                      # 动量
        # "weight_decay": 0.0005,               # 权重衰减
        # "lambda1": 0.001                      # 损失函数超参数            
        # }

        # _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)
        # # 初始化核心模型
        # self.prompt_ad = PromptAD(args).to(device)
        ###########################

        self.out_size_h = 400
        self.out_size_w = 400
        self.class_name = category
        # print("self.class_name: ",)
        self.k_shot = k_shot

        self.shot = k_shot
        self.precision = "fp16"
        # self.get_model(n_ctx=4, n_pro=3, n_ctx_ab=1, n_pro_ab=4, classname=category, 
        #                 backbone="ViT-B-16-plus-240", pretrained_dataset="laion400m_e32")
        #                 # backbone="ViT-L-14-336", pretrained_dataset="openai")
        # self.phrase_form = '{}'
        # self.device = device

        # self.transform = transforms.Compose([
        #     transforms.Resize((240, 240), Image.BICUBIC),
        #     transforms.CenterCrop(240),
        #     _convert_to_rgb,
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_train, std=std_train)])

        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((240, 240), Image.NEAREST),
        #     transforms.CenterCrop(240),
        #     transforms.ToTensor()])
        # self.transform = transforms.Compose([
        #     transforms.Resize((336, 336), Image.BICUBIC),
        #     transforms.CenterCrop(336),
        #     _convert_to_rgb,
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_train, std=std_train)])

        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((336, 336), Image.NEAREST),
        #     transforms.CenterCrop(336),
        #     transforms.ToTensor()])
        #######################################
        print("try to get seed when setup: ", torch.initial_seed())
        args = {
        "dataset": "mvtecloco",                # 数据集名称，可选['mvtec', 'visa', 'mvtecloco']
        "train_data": images,                   # 直接加载训练数据
        "class_name": category,                # 类别名称
        "img_resize": 336, # 240,                     # 图像缩放尺寸 ViT-B-16-plus-240
        "img_cropsize": 336, # 240,                   # 图像裁剪尺寸 ViT-B-16-plus-240
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
        "backbone": "ViT-L-14-336", # "ViT-B-16-plus-240",       # 骨干网络，可选['ViT-B-16-plus-240', 'ViT-B-16', ...]
        "pretrained_dataset": "openai", # "laion400m_e32", # 预训练数据集名称
        "use_cpu": 0,                         # 是否使用 CPU（0=否）
        "n_ctx": 4,                           # 上下文提示词数量
        "n_ctx_ab": 1,                        # 异常检测上下文提示词数量
        "n_pro": 3,                           # 原型数量
        "n_pro_ab": 4,                        # 异常检测原型数量
        "Epoch": 50,                         # 训练轮数
        "lr": 0.02,                          # 学习率
        "momentum": 0.9,                      # 动量
        "weight_decay": 0.0005,               # 权重衰减
        "lambda1": 0.001                      # 损失函数超参数            
        }
        
        main(args)

        pass

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, backbone, pretrained_dataset):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx=4, n_pro=3, n_ctx_ab=1, n_pro_ab=4, 
                                            classname=classname, clip_model=model, pre='fp16')
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

   

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        self.feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1))

        b2, n2, d2 = features2.shape
        self.feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0].cpu()
            print("self.text_features.T.device: ", self.text_features.T.device)
            print("global_feature.device: ", global_feature.device)
            # input()
            global_normality_and_abnormality_score = (t.cpu() * global_feature @ self.text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        visual_features = [f.cpu() for f in visual_features]

        N = visual_features[1].shape[0]

        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)


    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    

    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        
        # model, _, _ = CLIPAD.create_model_and_transforms(model_name='ViT-B-16-plus-240', pretrained='laion400m_e32', precision = 'fp16')
        # model, _, _ = CLIPAD.create_model_and_transforms(model_name='ViT-L-14-336', pretrained='openai', precision = 'fp16')
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

        # # print("image.shape: ", image.shape)
        # # for f in image:
        # #     print("f.shape: ", f.shape)
        # data = [
        # self.transform(
        #     Image.fromarray(
        #         (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))) 
        #         for f in image]
        # data = torch.stack(data, dim=0).to(self.device)
        # # print("image.shape after transform: ", data.shape)
        # # input()
        # # print("data.device: ", data.device)
        # visual_features = self.encode_image(data)
        # # print("visual_features device: ", visual_features.device)
        # # input()
        # textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

        # visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

        # anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
        #                             align_corners=False)

        # am_pix = anomaly_map.squeeze(1).numpy()

        # am_pix_list = []

        # for i in range(am_pix.shape[0]):
        #     am_pix_list.append(am_pix[i])

        # am_img_list = []
        # for i in range(textual_anomaly.shape[0]):
        #     am_img_list.append(textual_anomaly[i])

        # # return am_img_list, am_pix_list
        # pred_score = torch.tensor(am_img_list, device=image.device)
        
        ##################################

        kwargs = {
        "dataset": "mvtecloco",                # 数据集名称，可选['mvtec', 'visa', 'mvtecloco']
        "train_data": image,                   # 直接加载训练数据
        "class_name": self.class_name,                # 类别名称
        "img_resize": 336, # 240,                     # 图像缩放尺寸 ViT-B-16-plus-240
        "img_cropsize": 336, # 240,                   # 图像裁剪尺寸 ViT-B-16-plus-240
        "resolution": 400,                     # 分辨率
        "batch_size": 400,                     # 批量大小
        "vis": True,                           # 是否可视化（布尔值）
        "root_dir": "./src/eval/submission/result",                # 结果保存根目录
        "load_memory": True,                   # 是否加载到内存（布尔值）
        "cal_pro": False,                      # 是否计算概率（布尔值）
        # "seed": 111,                          # 随机种子
        "gpu_id": 3,                           # GPU ID
        "pure_test": False,                    # 是否仅测试模式（布尔值）
        "k_shot": self.k_shot,                      # Few-Shot 的样本数量
        "backbone": "ViT-L-14-336", # "ViT-B-16-plus-240",       # 骨干网络，可选['ViT-B-16-plus-240', 'ViT-B-16', ...]
        "pretrained_dataset": "openai", # "laion400m_e32", # 预训练数据集名称
        "use_cpu": 0,                         # 是否使用 CPU（0=否）
        "n_ctx": 4,                           # 上下文提示词数量
        "n_ctx_ab": 1,                        # 异常检测上下文提示词数量
        "n_pro": 3,                           # 原型数量
        "n_pro_ab": 4,                        # 异常检测原型数量
        "Epoch": 1,                         # 训练轮数
        "lr": 0.02,                          # 学习率
        "momentum": 0.9,                      # 动量
        "weight_decay": 0.0005,               # 权重衰减
        "lambda1": 0.001,                      # 损失函数超参数
        "out_size_h": 400,
        "out_size_w": 400,
        "device": f"cuda:0",            
        }

        device = f"cuda:0"
        print("self.category: ", self.class_name)
        # input()
        _, _, check_path = get_dir_from_args('CLS', **kwargs)
        
        model = PromptAD(**kwargs)
        model.to(device)
        model.eval_mode()
        model.load_state_dict(torch.load(check_path), strict=False)
        
        data = [
        model.transform(
            Image.fromarray(
                (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))) 
                for f in image]
        data = torch.stack(data, dim=0).to(device)

        score_img, score_map = model(data, 'cls')
        pred_score = torch.tensor(score_img, device=image.device)
        
        return ImageBatch(
            image=image,
            # pred_score=torch.zeros(batch_size, device=image.device),
            pred_score = pred_score
        )
