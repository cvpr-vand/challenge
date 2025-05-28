"""Model for submission."""
from anomalib.data import ImageBatch
import torch
from torch import nn
from torchvision.transforms import v2
import os
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F
from . import open_clip_local as open_clip
from scipy.optimize import linear_sum_assignment
import cv2
from .prompt_ensemble import encode_obj_text
from kmeans_pytorch import kmeans, kmeans_predict
from scipy.optimize import linear_sum_assignment
from .segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
from ultralytics import FastSAM

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.spatial.distance import mahalanobis

from .advancedinspector import AdvancedInspector

inspector = AdvancedInspector()

class FastSAMSegmenterFormatted:
    def __init__(self, device='cuda'):
        self.device = device
        try:
            self.model_fastsam = FastSAM('FastSAM-x.pt') # 或 'fastsam-s.pt'
            # model_fastsam 本身会自动移到可用设备，但可以明确指定
            # self.model_fastsam.to(self.device) # ultralytics 模型通常在推理时处理设备
            # print("Successfully loaded FastSAM model.")
        except Exception as e:
            print(f"下载或加载 FastSAM 权重失败: {e}")
            self.model_fastsam = None

    def generate_masks_formatted(self, raw_image_np_rgb, class_name):
        if self.model_fastsam is None:
            print("FastSAM model not loaded.")
            return []

        height, width = raw_image_np_rgb.shape[:2]
        
        if class_name == 'pushpins':
            results = self.model_fastsam(raw_image_np_rgb,
                                     device=self.device,
                                     retina_masks=True,
                                     imgsz=2048,#height, # 使用原始高度，或自定义
                                     conf=0.25,
                                     iou=0.7,
                                     verbose=False)
        elif class_name == 'screw_bag':
            results = self.model_fastsam(raw_image_np_rgb,
                                     device=self.device,
                                     retina_masks=True,
                                     imgsz=1024,#height, # 使用原始高度，或自定义
                                     conf=0.25,
                                     iou=0.7,
                                     verbose=False)
        else:
            results = self.model_fastsam(raw_image_np_rgb,
                                     device=self.device,
                                     retina_masks=True,
                                     imgsz=height, # 使用原始高度，或自定义
                                     conf=0.25,
                                     iou=0.7,
                                     verbose=False)
        
        formatted_masks = []
        if results and results[0].masks is not None:
            masks_tensor = results[0].masks.data.cpu().numpy() # (N, H, W)
            # xyxyn 是归一化的 [x1, y1, x2, y2]
            # xywhn 是归一化的 [x_center, y_center, width, height]
            # xyxy 是绝对像素的 [x1, y1, x2, y2]
            # xywh 是绝对像素的 [x_center, y_center, width, height]
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy() # (N, 4)
            confs = results[0].boxes.conf.cpu().numpy()    # (N,)

            for i in range(masks_tensor.shape[0]):
                mask_np = masks_tensor[i].astype(bool)
                area = int(mask_np.sum())
                
                x1, y1, x2, y2 = boxes_xyxy[i]
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)] # x, y, w, h

                formatted_masks.append({
                    'segmentation': mask_np,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': float(confs[i]), # 使用置信度作为 predicted_iou
                    'point_coords': [[(bbox[0] + bbox[2] // 2), (bbox[1] + bbox[3] // 2)]], # 使用 bbox 中心作为示例点
                    'stability_score': float(confs[i]), # 使用置信度作为 stability_score 的近似值
                    'crop_box': [0, 0, width, height] # FastSAM (ultralytics) 通常在全图操作
                })
        return formatted_masks

def to_np_img(m):
    m = m.permute(1, 2, 0).cpu().numpy()
    mean = np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    std = np.array([[[0.26862954, 0.26130258, 0.27577711]]])
    m  = m * std + mean
    return np.clip((m * 255.), 0, 255).astype(np.uint8)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Model(nn.Module):
    """TODO: Implement your model here"""
    def __init__(self) -> None:
        super().__init__()

        setup_seed(42)
        # NOTE: Create your transformation pipeline (if needed).
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = v2.Compose(
            [
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )
       
        local_model_path = "/home/dancer/LogSAD/clip_vitl14_model/open_clip_pytorch_model.bin"
        remote_model_id = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
        model_name = "ViT-L-14"
        if os.path.exists(local_model_path):
            print("Loading CLIP model from local checkpoint...")
            self.model_clip, _, _ = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=local_model_path
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            print("Local model not found. Loading CLIP model from HuggingFace Hub...")
            self.model_clip, _, _ = open_clip.create_model_and_transforms(remote_model_id)
            self.tokenizer = open_clip.get_tokenizer(remote_model_id)


        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        # # --- 策略3：使用更小的SAM模型 ---
        # SAM_MODEL_TYPE = "vit_h" #  "vit_h"、"vit_b"
        # SAM_CHECKPOINT_URL_BASE = "https://dl.fbaipublicfiles.com/segment_anything/"
        # SAM_CHECKPOINTS = {
        #     "vit_h": "sam_vit_h_4b8939.pth",
        #     "vit_l": "sam_vit_l_0b3195.pth",
        #     "vit_b": "sam_vit_b_01ec64.pth",
        # }
        # checkpoint_url = SAM_CHECKPOINT_URL_BASE + SAM_CHECKPOINTS[SAM_MODEL_TYPE]
        # self.model_sam = sam_model_registry[SAM_MODEL_TYPE]()
        # try:
        #     state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location=self.device)
        #     self.model_sam.load_state_dict(state_dict)
        #     # print(f"Successfully loaded SAM model: {SAM_MODEL_TYPE}")
        # except Exception as e:
        #     print(f"下载或加载 SAM ({SAM_MODEL_TYPE}) 权重失败: {e}")
        #     print("请检查网络连接、URL或本地权重路径是否正确。")
        # self.model_sam.to(self.device).eval()
        # # self.mask_generator = SamAutomaticMaskGenerator(model = self.model_sam)
        # # --- 策略2：调整SamAutomaticMaskGenerator参数 ---
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=self.model_sam,
        #     points_per_side=16,  # 默认 32。减少采样点数，显著加快速度。可以尝试 16, 8。
        #     # pred_iou_thresh=0.88, # 默认。略微提高 (如 0.9) 可以减少低质量掩码。
        #     # stability_score_thresh=0.95, # 默认。略微提高 (如 0.97) 可以减少不稳定掩码。
        #     # box_nms_thresh=0.7, # 默认。
        #     # min_mask_region_area=100,  # 默认 0。设置一个合理的最小面积（例如根据您的目标对象大小）可以过滤掉很多小噪声掩码。
        #     # points_per_batch=64, # 默认。如果显存足够，可以尝试增大；如果显存不足，可以减小。对总时间影响可能不大。
        #     # crop_n_layers=0, # 默认。 设为0表示不使用裁剪预测，可能会更快。
        #     # crop_n_points_downscale_factor=1 # 默认。
        # )

        # 替换为fastsam
        self.segmenter = FastSAMSegmenterFormatted(device=self.device)

        self.memory_size = 2048
        self.n_neighbors = 2

        self.model_clip.eval()
        self.test_args = None
        self.align_corners = True # False
        self.antialias = True # False
        self.inter_mode = 'bilinear' # bilinear/bicubic 
        
        self.cluster_feature_id = [0, 1]

        self.cluster_num_dict = {
            "breakfast_box": 3, # unused
            "juice_bottle": 8, # unused
            "splicing_connectors": 10, # unused
            "pushpins": 10, 
            "screw_bag": 10,
        }
        self.query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box', 'black background'],
            "juice_bottle": ['bottle', ['black background', 'background']],
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['screw'], 'plastic bag', 'background'],
            "splicing_connectors": [['splicing connector', 'splice connector',], ['cable', 'wire'], ['grid']],
        }
        self.foreground_label_idx = {  # for query_words_dict
            "breakfast_box": [0, 1, 2, 3, 4, 5],
            "juice_bottle": [0],
            "pushpins": [0],
            "screw_bag": [0], 
            "splicing_connectors":[0, 1]
        }

        self.patch_query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box', 'black background'],
            "juice_bottle": [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'], ['black background', 'background']], 
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['hex screw', 'hexagon bolt'], ['hex nut', 'hexagon nut'], ['ring washer', 'ring gasket'], ['plastic bag', 'background']],
            "splicing_connectors": [['splicing connector', 'splice connector',], ['cable', 'wire'], ['grid']],
        }
        

        self.query_threshold_dict = {
            "breakfast_box": [0., 0., 0., 0., 0., 0., 0.], # unused
            "juice_bottle": [0., 0., 0.], # unused
            "splicing_connectors": [0.15, 0.15, 0.15, 0., 0.], # unused
            "pushpins": [0.2, 0., 0., 0.],
            "screw_bag": [0., 0., 0.,],
        }

        self.feat_size = 64
        self.ori_feat_size = 32

        self.visualization = False

        self.pushpins_count = 15
        self.mem_traditional_features = None
        self.mem_traditional_stats = {} # 存储均值和协方差逆矩阵
        self.traditional_feature_dim = None

        self.splicing_connectors_count = [2, 3, 5] # coresponding to yellow, blue, and red
        self.splicing_connectors_distance = 0
        self.splicing_connectors_cable_color_query_words_dict = [['yellow cable', 'yellow wire'], ['blue cable', 'blue wire'], ['red cable', 'red wire']]
        
        self.juice_bottle_liquid_query_words_dict = [['red liquid', 'cherry juice'], ['yellow liquid', 'orange juice'], ['milky liquid']]
        self.juice_bottle_fruit_query_words_dict = ['cherry', ['tangerine', 'orange'], 'banana'] 

        # query words
        self.foreground_pixel_hist = 0  
        # patch query words
        self.patch_token_hist = []

        self.few_shot_inited = False


        from .dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        current_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(current_dir, "memory_bank", "statistic_scores_model_ensemble_few_shot_val.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Required model statistics file is missing: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.stats = pickle.load(f)
        pkl_path_tradional = os.path.join(current_dir, "memory_bank", "statistic_scores_with_traditional.pkl")
        if not os.path.exists(pkl_path_tradional):
            raise FileNotFoundError(f"Required model statistics file is missing: {pkl_path_tradional}")
        with open(pkl_path_tradional, "rb") as f:
            self.traditional_stats = pickle.load(f)
        pkl_path_threshold = os.path.join(current_dir, "memory_bank", "statistic_thresholds.pkl")
        if not os.path.exists(pkl_path_threshold):
            raise FileNotFoundError(f"Required model statistics file is missing: {pkl_path_threshold}")
        with open(pkl_path_threshold, "rb") as f:
            self.threshold_stats = pickle.load(f)
        pkl_path_threshold2 = os.path.join(current_dir, "memory_bank", "statistic_thresholds2.pkl")
        if not os.path.exists(pkl_path_threshold2):
            raise FileNotFoundError(f"Required model statistics file is missing: {pkl_path_threshold2}")
        with open(pkl_path_threshold2, "rb") as f:
            self.threshold_stats2 = pickle.load(f)
            
        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False #True #False

    def set_viz(self, viz):
        self.visualization = viz

    def set_val(self, val):
        self.validation = val

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        # pass
        few_shot_samples = setup_data.get("few_shot_samples")
        class_name = setup_data.get("dataset_category")
        self.class_name = class_name

        self.k_shot = few_shot_samples.size(0)
        self.process(class_name, few_shot_samples)
        self.few_shot_inited = True

    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        return None

    def forward(self, image: torch.Tensor) -> ImageBatch:
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): The input image batch of shape [B, 3, H, W].

        Returns:
            ImageBatch: The output image batch with prediction scores.
        """
        self.anomaly_flag = False

        # Ensure 4D input: [B, 3, H, W]
        if image.ndim == 3:  # single image, [3, H, W]
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        batch = image.clone().detach().to(self.device)
        batch = self.transform(batch)

        pred_scores = []
        if self.validation:
            hist_scores = []
            structural_scores = []
            instance_scores = []
            for i in range(batch_size):
                results = self.forward_one_sample(batch[i].unsqueeze(0), self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset)
                hist_scores.append(results['hist_score'])
                structural_scores.append(results['structural_score'])
                instance_scores.append(results['instance_hungarian_match_score'])

            return {
                "hist_score": torch.tensor(hist_scores, device=image.device),
                "structural_score": torch.tensor(structural_scores, device=image.device),
                "instance_hungarian_match_score": torch.tensor(instance_scores, device=image.device),
            }

        # Non-validation: perform scoring and sigmoid
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for i in range(batch_size):
            results = self.forward_one_sample(batch[i].unsqueeze(0), self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset)

            hist_score = results['hist_score']
            structural_score = results['structural_score']
            instance_hungarian_match_score = results['instance_hungarian_match_score']
            if self.class_name == 'pushpins':
                traditional_anomaly_score = results['traditional_anomaly_score'] # 获取新分数
                stats_trad = self.traditional_stats[self.class_name]["traditional_anomaly_scores"]
                mean_trad = stats_trad["mean"]
                std_trad = stats_trad["unbiased_std"]
                # 增加一个保护，防止标准差为0
                if std_trad > 1e-6:
                    standard_traditional_score = (traditional_anomaly_score - mean_trad) / std_trad
                else: # 如果标准差为0，只做中心化
                    standard_traditional_score = traditional_anomaly_score - mean_trad
                
                list_of_scores = results['list_of_scores']

            # standardization
            standard_structural_score = (
                structural_score - self.stats[self.class_name]["structural_scores"]["mean"]
            ) / self.stats[self.class_name]["structural_scores"]["unbiased_std"]
            standard_instance_hungarian_match_score = (
                instance_hungarian_match_score - self.stats[self.class_name]["instance_hungarian_match_scores"]["mean"]
            ) / self.stats[self.class_name]["instance_hungarian_match_scores"]["unbiased_std"]

            if self.class_name == 'pushpins':
                pred_score = max(standard_instance_hungarian_match_score, standard_structural_score, standard_traditional_score)
                # # 结合阈值进行判断
                # reasons = [] # 用于记录异常原因，便于调试
                # for score_name, score_value in list_of_scores.items():
                #     if score_name == 'head_area_score':
                #         threshold_info = self.threshold_stats2[score_name]
                #     else:
                #         threshold_info = self.threshold_stats[score_name]
                #     threshold = threshold_info['threshold']
                #     # 对污染使用您在inspector中定义的固定像素阈值
                #     # if score_name == 'contamination_score':
                #     #     if score_value > inspector.CONTAMINATION_PIXEL_THRESHOLD:
                #     #         self.anomaly_flag = True
                #     #         reasons.append(f"{score_name} | 计数值:{score_value} > 阈值:{inspector.CONTAMINATION_PIXEL_THRESHOLD}")
                #     # 其他分数都是值越大越异常
                #     # elif score_name == 'head_shape_score':
                #     #     if score_value > threshold:
                #     #         self.anomaly_flag = True
                #     #         reasons.append(f"{score_name} | 值:{score_value:.4f} > 阈值:{threshold:.4f}")
                #     if score_name == 'head_area_score':
                #         if score_value > threshold:
                #             self.anomaly_flag = True
                #             reasons.append(f"{score_name} | 值:{score_value:.4f} > 阈值:{threshold:.4f}")
            else:
                pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
            pred_score = sigmoid(pred_score)

            if self.anomaly_flag:
                pred_score = 1.0
                # if reasons:
                #     print(f"检测到异常！原因: {', '.join(reasons)}")
                self.anomaly_flag = False

            pred_scores.append(pred_score)

        return ImageBatch(
            image=image,
            pred_score=torch.tensor(pred_scores, device=image.device),
        )


    # def forward(self, image: torch.Tensor) -> ImageBatch:
    #     """Forward pass of the model.

    #     Args:
    #         image (torch.Tensor): The input image.

    #     Returns:
    #         ImageBatch: The output image batch.
    #     """
    #     # TODO: Implement the forward pass of the model.
    #     # Ensure 4D input: [B, 3, H, W]
    #     if image.ndim == 3:  # single image, [3, H, W]
    #         image = image.unsqueeze(0)
    #     batch_size = image.shape[0]
    #     if batch_size > 1:
    #         raise RuntimeError("out of memory")

    #     self.anomaly_flag = False
    #     batch = image.clone().detach().to(self.device)
    #     batch = self.transform(batch)
    #     results = self.forward_one_sample(batch, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset)

    #     hist_score = results['hist_score']
    #     structural_score = results['structural_score']
    #     instance_hungarian_match_score = results['instance_hungarian_match_score']

    #     anomaly_map_structural = results['anomaly_map_structural']

    #     if self.validation:
    #         return {"hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score), "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}

    #     def sigmoid(z):
    #         return 1/(1 + np.exp(-z))
        
    #     # standardization
    #     standard_structural_score = (structural_score - self.stats[self.class_name]["structural_scores"]["mean"]) / self.stats[self.class_name]["structural_scores"]["unbiased_std"]
    #     standard_instance_hungarian_match_score = (instance_hungarian_match_score - self.stats[self.class_name]["instance_hungarian_match_scores"]["mean"]) / self.stats[self.class_name]["instance_hungarian_match_scores"]["unbiased_std"]
 
    #     pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
    #     pred_score = sigmoid(pred_score)
        
    #     if self.anomaly_flag:
    #         pred_score = 1.
    #         self.anomaly_flag = False

    #     # return {"pred_score": torch.tensor(pred_score), "anomaly_map": torch.tensor(anomaly_map_structural), "hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score), "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}

    #     return ImageBatch(
    #         image=image,
    #         pred_score=torch.tensor(pred_score).to(image.device),
    #     )
    
    def forward_one_sample(self, batch: torch.Tensor, mem_patch_feature_clip_coreset: torch.Tensor, mem_patch_feature_dinov2_coreset: torch.Tensor, path: str = ""):
        batch = F.interpolate(batch, size=(448, 448), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        
        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(batch, self.feature_list)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (1, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (1, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1) # (1x64x64, 1024x4)
        
        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(batch, out_layer_list=self.feature_list)
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1) # (1x64x64, 1024x4)
        
        '''adding for kmeans seg '''
        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(self.feat_size * self.feat_size, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        mid_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            mid_features = temp_feat if mid_features is None else torch.cat((mid_features, temp_feat), -1)
            
        if self.feat_size != self.ori_feat_size:
            mid_features = mid_features.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            mid_features = F.interpolate(mid_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            mid_features = mid_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        mid_features = F.normalize(mid_features, p=2, dim=-1)
             
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name, os.path.dirname(path).split('/')[-1] + "_" + os.path.basename(path).split('.')[0])
        
        hist_score = results['score']

        '''calculate patchcore'''
        anomaly_maps_patchcore = []

        if self.class_name in ['pushpins', 'screw_bag']: # clip feature for patchcore
            len_feature_list = len(self.feature_list)
            for patch_feature, mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                if self.class_name == 'pushpins':
                    k_topk = 6
                else:
                    k_topk = 3
                topk_vals, _ = normal_map_patchcore.topk(k=k_topk, dim=1)
                normal_map_patchcore = topk_vals.mean(1).cpu().numpy()
                # normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal 
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        if self.class_name in ['splicing_connectors', 'breakfast_box', 'juice_bottle']: # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for patch_feature, mem_patch_feature in zip(patch_tokens_dinov2.chunk(len_feature_list, dim=-1), mem_patch_feature_dinov2_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                if self.class_name == 'splicing_connectors':
                    k_topk = 5
                else:
                    k_topk = 3
                topk_vals, _ = normal_map_patchcore.topk(k=k_topk, dim=1)
                normal_map_patchcore = topk_vals.mean(1).cpu().numpy()
                # normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal   
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        # --- 新增：计算模板匹配分数 ---
        if self.class_name == 'pushpins' and 'binary_foreground' in results:
            np_image = to_np_img(batch[0])
            foreground_mask = results['binary_foreground']
            list_of_scores = inspector.predict(np_image, foreground_mask)
        
        # --- 新增：计算传统特征异常分数 ---
        traditional_anomaly_score = 0.0 # 默认为正常
        if self.class_name == 'pushpins' and 'binary_foreground' in results and self.mem_traditional_features is not None:
            np_image = to_np_img(batch[0])
            foreground_mask = results['binary_foreground']
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            debug_image_name = os.path.join(debug_dir, "debug_masks.png")
            # 将掩码从 0/1 变为 0/255 的灰度图
            cv2.imwrite(debug_image_name, foreground_mask * 255)
            
            # 1. 为测试样本提取传统特征
            test_features = self.extract_traditional_features_v2(np_image, foreground_mask)
            if test_features is None:
                test_features = np.zeros(self.traditional_feature_dim)
            
            # 2. 计算马氏距离
            # 如果特征维度不匹配（例如测试时未检测到物体），给一个最大的惩罚分数
            if test_features.shape[0] != self.mem_traditional_stats['mean'].shape[0]:
                traditional_anomaly_score = 10.0 # 一个比较大的值
            else:
                try:
                    # 使用存储的均值和逆协方差矩阵
                    mean_ff = self.mem_traditional_stats['mean']
                    # 【核心修改】根据是否存在 'cov_inv' 来选择距离度量
                    if 'cov_inv' in self.mem_traditional_stats:
                        # k > 1 的情况：使用马氏距离
                        cov_inv_ff = self.mem_traditional_stats['cov_inv']
                        traditional_anomaly_score = mahalanobis(test_features, mean_ff, cov_inv_ff)
                    else:
                        # k = 1 的情况：降级为使用余弦距离或欧氏距离
                        # 选项 A: 余弦距离 (推荐)
                        similarity = (test_features @ mean_ff) / (np.linalg.norm(test_features) * np.linalg.norm(mean_ff) + 1e-6)
                        traditional_anomaly_score = 1.0 - similarity
                        # 选项 B: 欧氏距离
                        # traditional_anomaly_score = np.linalg.norm(test_features - mean_ff)
                except Exception as e:
                    print(f"Error calculating Mahalanobis distance: {e}")
                    traditional_anomaly_score = 10.0

        structural_score = np.stack(anomaly_maps_patchcore).mean(0).max()
        anomaly_map_structural = np.stack(anomaly_maps_patchcore).mean(0).reshape(self.feat_size, self.feat_size)

        instance_masks = results["instance_masks"] 
        anomaly_instances_hungarian = []
        instance_hungarian_match_score = 1.
        if self.mem_instance_masks is not None and len(instance_masks) != 0:
            for patch_feature, batch_mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                instance_features = [patch_feature[mask, :].mean(0, keepdim=True) for mask in instance_masks]
                instance_features = torch.cat(instance_features, dim=0)
                instance_features = F.normalize(instance_features, dim=-1)
                mem_instance_features = []
                for mem_patch_feature, mem_instance_masks in zip(batch_mem_patch_feature.chunk(self.k_shot), self.mem_instance_masks):
                    mem_instance_features.extend([mem_patch_feature[mask, :].mean(0, keepdim=True) for mask in mem_instance_masks])
                mem_instance_features = torch.cat(mem_instance_features, dim=0)
                mem_instance_features = F.normalize(mem_instance_features, dim=-1)

                normal_instance_hungarian = (instance_features @ mem_instance_features.T)
                cost_matrix = (1 - normal_instance_hungarian).cpu().numpy()
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                cost = cost_matrix[row_ind, col_ind].sum() 
                cost = cost / min(cost_matrix.shape)
                anomaly_instances_hungarian.append(cost)

            instance_hungarian_match_score = np.mean(anomaly_instances_hungarian)     

        if self.class_name == 'pushpins':
            results = {'hist_score': hist_score, 'structural_score': structural_score,
                        'traditional_anomaly_score': traditional_anomaly_score,
                          'instance_hungarian_match_score': instance_hungarian_match_score,
                            "anomaly_map_structural": anomaly_map_structural,
                            "list_of_scores": list_of_scores}
        else:
            results = {'hist_score': hist_score, 'structural_score': structural_score,  'instance_hungarian_match_score': instance_hungarian_match_score, "anomaly_map_structural": anomaly_map_structural}

        return results


    def histogram(self, image, cluster_feature, proj_patch_token, class_name, path):
        def plot_results_only(sorted_anns):
            cur = 1
            img_color = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            for ann in sorted_anns:
                m = ann['segmentation']
                img_color[m] = cur
                cur += 1
            return img_color
        
        def merge_segmentations(a, b, background_class):
            unique_labels_a = np.unique(a)
            unique_labels_b = np.unique(b)

            max_label_a = unique_labels_a.max()
            label_map = np.zeros(max_label_a + 1, dtype=int)

            for label_a in unique_labels_a:
                mask_a = (a == label_a)

                labels_b = b[mask_a]
                if labels_b.size > 0:
                    count_b = np.bincount(labels_b, minlength=unique_labels_b.max() + 1)
                    label_map[label_a] = np.argmax(count_b)
                else:
                    label_map[label_a] = background_class # default background

            merged_a = label_map[a]
            return merged_a
        
        pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device)
        kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)    # default to background
        
        for pl in pseudo_labels.unique():
            mask = (pseudo_labels == pl).reshape(-1)
            # filter small region
            binary = mask.cpu().numpy().reshape(self.feat_size, self.feat_size).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 8:
                    mask[temp_mask.reshape(-1)] = False

            if mask.any():
                region_feature = proj_patch_token[mask, :].mean(0, keepdim=True)
                similarity = (region_feature @ self.query_obj.T)
                prob, index = torch.max(similarity, dim=-1)
                temp_label = index.squeeze(0).item()
                temp_prob = prob.squeeze(0).item()
                if temp_prob > self.query_threshold_dict[class_name][temp_label]: # threshold for each class
                    kmeans_mask[mask] = temp_label    


        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        # if self.class_name == 'splicing_connectors':
        #     masks = self.mask_generator.generate(raw_image)
        # else:
        #     masks = self.segmenter.generate_masks_formatted(raw_image, self.class_name)
        masks = self.segmenter.generate_masks_formatted(raw_image, self.class_name)
        # self.predictor.set_image(raw_image)
        masks_ori = masks.copy()
        if self.class_name == 'pushpins':
            grid_masks = []
            for mask_info in masks_ori[0:]:
                bottom_forbidden_zone = {
                    "x_min": 0,
                    "x_max": width,
                    "y_min": height * 0.93,
                    "y_max": height 
                }
                if mask_info['area'] < 1200: 
                    continue
                mask = mask_info['segmentation'].astype(np.uint8) * 255
                x, y, w, h = mask_info['bbox']
                center_x = x + w / 2
                center_y = y + h / 2
                if (bottom_forbidden_zone["x_min"] < center_x < bottom_forbidden_zone["x_max"] and
                    bottom_forbidden_zone["y_min"] < center_y < bottom_forbidden_zone["y_max"]):
                    continue
                # 缩小 bbox 到中心区域（缩一半）
                cx, cy = x + w // 4, y + h // 4
                cw, ch = w // 2, h // 2
                center_mask = mask[cy:cy+ch, cx:cx+cw]
                # 提取图像中心区域
                bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                center_image = bgr_image[cy:cy+ch, cx:cx+cw]  # bgr_image 是原图 BGR
                # 掩码区域内像素
                hsv_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                h_channel, s_channel, v_channel = cv2.split(hsv_image)
                # 在整个中心区域中查找符合条件的像素
                valid = np.logical_and.reduce([
                    h_channel >= 0, h_channel <= 50,
                    s_channel >= 0, s_channel <= 50,
                    v_channel >= 100, v_channel <= 200
                ])
                if np.sum(valid) > 50:
                    continue  # 过滤掉这种 mask
                grid_masks.append(mask_info)
        no_pushpins_detected = False
        if self.class_name == 'pushpins':
            # 默认假设检测到了图钉
            filtered_masks = []
            # 保留面积最大的第一个掩码（作为背景）
            # background_mask_info = sorted_masks[0]
            # filtered_masks.append(background_mask_info)
            # 遍历其余的掩码（从第二个元素开始）
            # --- 定义禁区 ---
            # 1. 顶部中间区域
            top_forbidden_zone = {
                "x_min": width * 0.4,
                "x_max": width * 0.6,
                "y_min": 0,
                "y_max": height * 0.1 
            }
            # 2. 左侧中间区域
            left_forbidden_zone = {
                "x_min": 0,
                "x_max": width * 0.05,
                "y_min": height * 0.48,
                "y_max": height * 0.56
            }
            forbidden_zones = [top_forbidden_zone, left_forbidden_zone]
            for mask_info in masks[0:]:
                # 只保留面积小于或等于 1000 的掩码
                if mask_info['area'] > 1000 or mask_info['area'] < 32:
                    continue 
                # --- 条件2: 宽高比筛选 ---
                bbox = mask_info['bbox']
                x, y, w, h = bbox
                # 计算长边与短边的比例
                aspect_ratio = max(w / h, h / w)
                # 如果宽高比过于极端，则跳过
                if aspect_ratio > 8.0:
                    continue # 跳过这个掩码
                # --- 位置筛选逻辑 ---
                is_in_forbidden_zone = False
                # 计算 bbox 的中心点
                center_x = x + w / 2
                center_y = y + h / 2
                for zone in forbidden_zones:
                    if (zone["x_min"] < center_x < zone["x_max"] and
                        zone["y_min"] < center_y < zone["y_max"]):
                        is_in_forbidden_zone = True
                        # print(f"剔除位于禁区的掩码: bbox={bbox}")
                        break # 一旦落入一个禁区，就无需再检查其他禁区
                if is_in_forbidden_zone:
                    continue # 跳过这个掩码
                # --- 如果所有条件都通过，则保留该掩码 ---
                filtered_masks.append(mask_info)

            if len(filtered_masks) == 0:
                no_pushpins_detected = True
            masks = filtered_masks
        
        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy()
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1) 
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        # sam_mask = plot_results_only(sorted_masks).astype(int)
        if no_pushpins_detected:
            sam_mask = np.zeros((height, width), dtype=np.int32)
        else:
            sam_mask = plot_results_only(sorted_masks).astype(int)
        
        resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        merge_sam = merge_segmentations(sam_mask, resized_mask, background_class=self.classes-1)
        merge_sam_noclip = np.where(sam_mask == 0, 1, 0)

        resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)
        
        # filter small region for merge sam
        binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32: # 448x448 
                merge_sam[temp_mask] = self.classes - 1 # set to background

        # filter small region for merge sam no clip
        binary_noclip = np.isin(merge_sam_noclip, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
        num_labels_noclip, labels_noclip, stats_noclip, centroids_noclip = cv2.connectedComponentsWithStats(binary_noclip, connectivity=8)
        for i in range(1, num_labels_noclip):
            temp_mask_noclip = labels_noclip == i
            if np.sum(temp_mask_noclip) <= 200: # 448x448 
                merge_sam_noclip[temp_mask_noclip] = self.classes - 1 # set to background

        # filter small region for patch merge sam
        binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32: # 448x448
                patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background

        score = 0. # default to normal
        self.anomaly_flag = False
        instance_masks = []
        if self.class_name == 'pushpins':
            # object count hist
            kernel = np.ones((3, 3), dtype=np.uint8)  # dilate for robustness  
            binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8) # foreground 1  background 0
            dilate_binary = cv2.dilate(binary, kernel)
            binary_noclip = np.isin(merge_sam_noclip, self.foreground_label_idx[self.class_name]).astype(np.uint8) # foreground 1  background 0
            dilate_binary_noclip = cv2.dilate(binary_noclip, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            # pushpins_count = 0
            # valid_centroids = []
            # for i in range(1, num_labels):
            #     if stats[i, cv2.CC_STAT_AREA] < 100: continue
            #     x, y, w, h, _ = stats[i]
            #     instance_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            #     instance_image = raw_image[y:y+h, x:x+w]
            #     # image = Image.fromarray(instance_image, mode='RGB')
            #     # image.save("special_variable_image.png")
            #     # debug_dir = "debug_images"
            #     # os.makedirs(debug_dir, exist_ok=True)
            #     # pushpin_masks_name = os.path.join(debug_dir, "pushpin_masks.png")
            #     # cv2.imwrite(pushpin_masks_name, instance_mask)
            #     instance_image = cv2.cvtColor(instance_image, cv2.COLOR_RGB2BGR)
            #     hsv_instance_only = cv2.bitwise_and(instance_image, instance_image, mask=instance_mask)
            #     hsv_instance = cv2.cvtColor(hsv_instance_only, cv2.COLOR_BGR2HSV)
            #     HEAD_YELLOW_LOWER = np.array([15, 180, 46])
            #     HEAD_YELLOW_UPPER = np.array([35, 255, 255])
            #     yellow_mask = cv2.inRange(hsv_instance, HEAD_YELLOW_LOWER, HEAD_YELLOW_UPPER)
            #     # debug_dir = "debug_images"
            #     # os.makedirs(debug_dir, exist_ok=True)
            #     # head_masks_name = os.path.join(debug_dir, "head_masks.png")
            #     # cv2.imwrite(head_masks_name, yellow_mask)
            #     # yellow_mask 中非零且 instance_mask 中非零的才是有效区域
            #     valid_pixels = np.logical_and(yellow_mask > 0, instance_mask > 0)
            #     # 5. 计算比例
            #     yellow_ratio = np.sum(valid_pixels) / np.sum(instance_mask > 0)
            #     # 6. 判断是否大多数为黄色（如大于 80%）
            #     if yellow_ratio > 0.5:
            #         pushpins_count += 1
            #         valid_centroids.append(centroids[i])
            pushpins_count = num_labels - 1 # number of pushpins
            if no_pushpins_detected:
                pushpins_count = 0

            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            if self.few_shot_inited and pushpins_count != self.pushpins_count and self.anomaly_flag is False:
                self.anomaly_flag = True
                print('number of pushpins: {}, but canonical number of pushpins: {}'.format(pushpins_count, self.pushpins_count))
            
            # ✅ 新增逻辑：图钉数量匹配但要检查每个格子最多一个图钉
            if pushpins_count == self.pushpins_count and self.anomaly_flag is False:
                # 图钉中心点列表
                pushpin_centers = centroids[1:]  # 去掉背景，第0个是背景
                # 15个格子 bbox 坐标
                grid_boxes = [
                    (186, 42, 77, 121),(0, 37, 95, 125),(0, 169, 95, 120),(0, 300, 95, 124),(185, 169, 77, 119),(354, 300, 93, 121),(353, 170, 74, 119),
                    (267, 302, 76, 117),(354, 38, 93, 124),(103, 170, 77, 118),(103, 38, 77, 124),(269, 171, 73, 117),(268, 37, 75, 125),(103, 300, 76, 117),
                    (186, 300, 76, 117)]
                # 统计每个格子里的图钉数量
                from collections import defaultdict
                grid_pushpin_counts = defaultdict(int)
                for (cx, cy) in pushpin_centers:
                    for i, (x, y, w, h) in enumerate(grid_boxes):
                        if x <= cx < x + w and y <= cy < y + h:
                            grid_pushpin_counts[i] += 1
                            break
                # 检查是否有格子包含多个图钉
                multiple_in_one_grid = [i for i, count in grid_pushpin_counts.items() if count > 1]
                if multiple_in_one_grid:
                    self.anomaly_flag = True
                    print("异常：以下格子含有多个图钉：", multiple_in_one_grid)

            # ✅ 新增逻辑：格子的面积不能大于 20000（实际效果受限于fastsam并不能准确分割出每个格子）
            if pushpins_count == self.pushpins_count and self.anomaly_flag is False:
                for mask_info in grid_masks:
                    x, y, w, h = mask_info['bbox']
                    area = w * h
                    if area > 20000:
                        self.anomaly_flag = True
                        print("anomaly: Missing_separator")
                        break

            # patch hist 
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # binary_foreground = dilate_binary.astype(np.uint8) 
            binary_foreground = dilate_binary_noclip.astype(np.uint8)

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

            # todo: same number in total but in different boxes or broken box
            return {"score": score, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks, "binary_foreground": binary_foreground}
        
        elif self.class_name == 'splicing_connectors':
            #  object count hist for default
            # # --- 策略1：降低SAM输入分辨率 ---
            # largest_mask_high_res_bool = (sam_mask == 1)
            # sam_mask_max_area = largest_mask_high_res_bool # background
            # sam_mask_max_area = sorted_masks[0]['segmentation'] # background
            # binary = (sam_mask_max_area == 0).astype(np.uint8) # sam_mask_max_area is background,  background 0 foreground 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 200: # 448x448 64
                    binary[temp_mask] = 0 # set to background

            kernel = np.ones((3, 3), dtype=np.uint8)  # dilate for robustness  
            dilate_binary = cv2.dilate(binary, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                # 获取该实例的掩码区域对应的图像块（假设有 raw_image）
                x, y, w, h, _ = stats[i]
                instance_mask_crop = temp_mask[y:y+h, x:x+w].astype(np.uint8) * 255
                instance_image_crop = raw_image[y:y+h, x:x+w]  
                instance_image_crop = cv2.cvtColor(instance_image_crop, cv2.COLOR_RGB2BGR)# BGR 图像
                hsv_instance_only = cv2.bitwise_and(instance_image_crop, instance_image_crop, mask=instance_mask_crop)
                hsv_instance = cv2.cvtColor(hsv_instance_only, cv2.COLOR_BGR2HSV)
                h_channel, s_channel, v_channel = cv2.split(hsv_instance)
                mask_indices = instance_mask_crop > 0
                s_values = s_channel[mask_indices]
                v_values = v_channel[mask_indices]
                s_thresh_ratio = np.sum(s_values <= 80) / len(s_values)
                v_thresh_ratio = np.sum(v_values <= 80) / len(v_values)
                # 如果大多数（85%）像素都是低饱和低亮度，认为是灰暗区域，过滤掉
                if s_thresh_ratio > 0.5 and v_thresh_ratio > 0.5:
                    binary[temp_mask] = 0
                    continue
                count += 1  # 保留的实例数量
            # count = num_labels - 1 # number of splicing connectors, -1 for background

            if count != 1 and self.anomaly_flag is False: # cable cut or no cable or no connector
                print('number of connected component in splicing_connectors: {}, but the default connected component is 1.'.format(count))
                self.anomaly_flag = True

            merge_sam[~(binary.astype(np.bool))] = self.query_obj.shape[0] - 1 # remove noise
            patch_merge_sam[~(binary.astype(np.bool))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            # erode the cable and divide into left and right parts
            kernel = np.ones((23, 23), dtype=np.uint8)
            erode_binary = cv2.erode(binary, kernel)
            h, w = erode_binary.shape
            distance = 0

            left, right = erode_binary[:, :int(w/2)],  erode_binary[:, int(w/2):]   
            left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
            right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1] # foreground

            binary_cable = (patch_merge_sam == 1).astype(np.uint8) 
            
            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_cable = cv2.erode(binary_cable, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cable, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448
                    binary_cable[temp_mask] = 0 # set to background
                

            binary_cable = cv2.resize(binary_cable, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

            binary_clamps = (patch_merge_sam == 0).astype(np.uint8)

            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_clamps = cv2.erode(binary_clamps, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clamps, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448
                    binary_clamps[temp_mask] = 0 # set to background
                else:
                    instance_mask = temp_mask.astype(np.uint8)
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                    if instance_mask.any():
                        instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            binary_clamps = cv2.resize(binary_clamps, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

            binary_connector = cv2.resize(binary, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            
            query_cable_color = encode_obj_text(self.model_clip, self.splicing_connectors_cable_color_query_words_dict, self.tokenizer, self.device)
            cable_feature = proj_patch_token[binary_cable.astype(np.bool).reshape(-1), :].mean(0, keepdim=True)
            idx_color = (cable_feature @ query_cable_color.T).argmax(-1).squeeze(0).item()
            foreground_pixel_count = np.sum(erode_binary) / self.splicing_connectors_count[idx_color]


            slice_cable = binary[:, int(w/2)-1: int(w/2)+1]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
            cable_count = num_labels - 1
            if cable_count != 1 and self.anomaly_flag is False: # two cables
                print('number of cable count in splicing_connectors: {}, but the default cable count is 1.'.format(cable_count))
                self.anomaly_flag = True

            # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:    # color and number mismatch
                    print('cable color and number of clamps mismatch, cable color idx: {} (0: yellow 2-clamp, 1: blue 3-clamp, 2: red 5-clamp), foreground_pixel_count :{}, canonical foreground_pixel_hist: {}.'.format(idx_color, foreground_pixel_count, self.foreground_pixel_hist))
                    self.anomaly_flag = True

            # left right hist for symmetry
            ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
            if self.few_shot_inited and (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False: # left right asymmetry in clamp
                print('left and right connectors are not symmetry.')
                self.anomaly_flag = True

            # left and right centroids distance
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode_binary, connectivity=8)
            if num_labels - 1 == 2:
                centroids = centroids[1:]
                x1, y1 = centroids[0] 
                x2, y2 = centroids[1]
                distance = np.sqrt((x1/w - x2/w)**2 + (y1/h - y2/h)**2)
                if self.few_shot_inited and self.splicing_connectors_distance != 0 and self.anomaly_flag is False:
                    ratio = distance / self.splicing_connectors_distance
                    if ratio < 0.6 or ratio > 1.4:  # too short or too long centroids distance (cable) # 0.6 1.4
                        print('cable is too short or too long.')
                        self.anomaly_flag = True

            # patch hist 
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])#[:-1]  # ignore background (grid) for statistic
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # todo    mismatch cable link  
            binary_foreground = binary.astype(np.uint8) # only 1 instance, so additionally seperate cable and clamps
            if binary_connector.any():
                instance_masks.append(binary_connector.astype(np.bool).reshape(-1))
            if binary_clamps.any():
                instance_masks.append(binary_clamps.astype(np.bool).reshape(-1))
            if binary_cable.any():
                instance_masks.append(binary_cable.astype(np.bool).reshape(-1))      

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, binary_connector, merge_sam, patch_merge_sam, erode_binary, binary_cable, binary_clamps]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'binary_connector', 'merge sam', 'patch merge sam', 'erode binary', 'binary_cable', 'binary_clamps']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "distance": distance, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'screw_bag':
            # pixel hist of kmeans mask
            foreground_pixel_count = np.sum(np.bincount(kmeans_mask.reshape(-1))[:len(self.foreground_label_idx[self.class_name])])  # foreground pixel
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                # todo: optimize
                if ratio < 0.94 or ratio > 1.06: 
                    print('foreground pixel histagram of screw bag: {}, the canonical foreground pixel histogram of screw bag in few shot: {}'.format(foreground_pixel_count, self.foreground_pixel_hist))
                    self.anomaly_flag = True

            # patch hist
            binary_screw = np.isin(kmeans_mask, self.foreground_label_idx[self.class_name])
            patch_mask[~binary_screw] = self.patch_query_obj.shape[0] - 1 # remove patch noise
            resized_binary_screw = cv2.resize(binary_screw.astype(np.uint8), (patch_merge_sam.shape[1], patch_merge_sam.shape[0]), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam[~(resized_binary_screw.astype(np.bool))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])[:-1]
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # # todo: count of screw, nut and washer, screw of different length
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()
            
            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'breakfast_box':
            # patch hist
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0]) 
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()
            
            # todo: exist of foreground

            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]
            
            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

            return {"score": score, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'juice_bottle': 
            # remove noise due to non sam mask
            merge_sam[sam_mask == 0] = self.classes - 1
            patch_merge_sam[sam_mask == 0] = self.patch_query_obj.shape[0] - 1  # 79.5

            # [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'], ['black background', 'background']], 
            # fruit and liquid mismatch (todo if exist)
            resized_patch_merge_sam = cv2.resize(patch_merge_sam, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            binary_liquid = (resized_patch_merge_sam == 1)
            binary_fruit = (resized_patch_merge_sam == 2)

            query_liquid = encode_obj_text(self.model_clip, self.juice_bottle_liquid_query_words_dict, self.tokenizer, self.device)
            query_fruit = encode_obj_text(self.model_clip, self.juice_bottle_fruit_query_words_dict, self.tokenizer, self.device)

            liquid_feature = proj_patch_token[binary_liquid.reshape(-1), :].mean(0, keepdim=True)
            liquid_idx = (liquid_feature @ query_liquid.T).argmax(-1).squeeze(0).item()

            fruit_feature = proj_patch_token[binary_fruit.reshape(-1), :].mean(0, keepdim=True)
            fruit_idx = (fruit_feature @ query_fruit.T).argmax(-1).squeeze(0).item()
            
            if (liquid_idx != fruit_idx) and self.anomaly_flag is False:
                print('liquid: {}, but fruit: {}.'.format(self.juice_bottle_liquid_query_words_dict[liquid_idx], self.juice_bottle_fruit_query_words_dict[fruit_idx]))
                self.anomaly_flag = True

            # # todo centroid of fruit and tag_0 mismatch (if exist) ,  only one tag, center

            # patch hist 
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:  
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T) 
                score = 1 - patch_hist_similarity.max()
            
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1) ).astype(np.uint8) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show() 

            return {"score": score, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}

        return {"score": score, "instance_masks": instance_masks}


    def process_k_shot(self, class_name, few_shot_samples):
        few_shot_samples = F.interpolate(few_shot_samples, size=(448, 448), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)

        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(few_shot_samples, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (bs, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (bs, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1)  # (bsx64x64, 1024x4)

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(few_shot_samples, out_layer_list=self.feature_list_dinov2)  # 4 x [bs, 32x32, 1024]
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (bs, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (bsx64x64, 1024x4)

        cluster_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            cluster_features = temp_feat if cluster_features is None else torch.cat((cluster_features, temp_feat), 1)
        if self.feat_size != self.ori_feat_size:
            cluster_features = cluster_features.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            cluster_features = F.interpolate(cluster_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            cluster_features = cluster_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        cluster_features = F.normalize(cluster_features, p=2, dim=-1)

        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        num_clusters = self.cluster_num_dict[class_name]
        _, self.cluster_centers = kmeans(X=cluster_features, num_clusters=num_clusters, device=self.device)
    
        self.query_obj = encode_obj_text(self.model_clip, self.query_words_dict[class_name], self.tokenizer, self.device)
        self.patch_query_obj = encode_obj_text(self.model_clip, self.patch_query_words_dict[class_name], self.tokenizer, self.device)
        self.classes = self.query_obj.shape[0]

        scores = []
        foreground_pixel_hist = []
        splicing_connectors_distance = []
        patch_token_hist = []
        mem_instance_masks = []
        mem_traditional_features_list = []

        few_shot_data = [] # 用于fewshot_setup创建模板   
        for image, cluster_feature, proj_patch_token in zip(
            few_shot_samples.chunk(self.k_shot), 
            cluster_features.chunk(self.k_shot), 
            proj_patch_tokens.chunk(self.k_shot)
        ):        
            # path = os.path.dirname(few_shot_path).split('/')[-1] + "_" + os.path.basename(few_shot_path).split('.')[0]
            self.anomaly_flag = False
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name, "")
            if self.class_name == 'pushpins':
                patch_token_hist.append(results["clip_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])
                # 从 histogram 的结果中提取前景掩码
                if 'binary_foreground' in results:
                    np_image = to_np_img(image[0])
                    foreground_mask = results['binary_foreground']
                    few_shot_data.append((np_image, foreground_mask))
                    # 提取传统特征
                    traditional_features = self.extract_traditional_features_v2(np_image, foreground_mask)
                    if traditional_features is not None:
                        # 如果维度还未设置，就用第一个成功提取的特征向量长度来设置它
                        if self.traditional_feature_dim is None:
                            self.traditional_feature_dim = len(traditional_features)
                    else:
                        traditional_features = np.zeros(self.traditional_feature_dim)
                    mem_traditional_features_list.append(traditional_features)

            elif self.class_name == 'splicing_connectors':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                splicing_connectors_distance.append(results["distance"])
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'screw_bag':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                patch_token_hist.append(results["clip_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'breakfast_box':
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'juice_bottle':
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            scores.append(results["score"])

        # 进行fewshot_setup
        if self.class_name == 'pushpins':
            inspector.setup(few_shot_data)

        if len(foreground_pixel_hist) != 0:
            self.foreground_pixel_hist = np.mean(foreground_pixel_hist)
        if len(splicing_connectors_distance) != 0:
            self.splicing_connectors_distance = np.mean(splicing_connectors_distance)
        if len(patch_token_hist) != 0: # patch hist
            self.patch_token_hist = np.stack(patch_token_hist)
        if len(mem_instance_masks) != 0:
            self.mem_instance_masks = mem_instance_masks

        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        # 在函数的最后，计算并存储传统特征的统计数据
        if mem_traditional_features_list:
            self.mem_traditional_features = np.array(mem_traditional_features_list)
            # 计算均值和协方差矩阵的逆，用于马氏距离
            mean = np.mean(self.mem_traditional_features, axis=0)
            self.mem_traditional_stats = {'mean': mean}
            # 只在样本数大于1时才计算协方差
            if self.mem_traditional_features.shape[0] > 1:
                # 添加一个小的对角矩阵以保证协方差矩阵是可逆的 (regularization)
                cov = np.cov(self.mem_traditional_features, rowvar=False)
                cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-5)#2) # 建议增加正则化强度
                self.mem_traditional_stats['cov_inv'] = cov_inv # 将协方差的逆添加到字典中

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset

    
    
    def process(self, class_name: str, few_shot_samples: list[torch.Tensor]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device)
        scores, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(class_name, few_shot_samples)

    def extract_traditional_features(self, img_rgb: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        """
        从给定的图像和前景掩码中提取传统CV特征向量。
        该函数处理多个分离的轮廓（例如多个图钉），并返回一个聚合的、固定长度的特征向量。

        Args:
            image_rgb (np.ndarray): rgb格式的输入图像 (H, W, 3), uint8.
            foreground_mask (np.ndarray): 前景掩码 (H, W), uint8, 0代表背景, >0代表前景.

        Returns:
            np.ndarray: 包含所有特征的、扁平化的一维Numpy数组。
        """
        # 0. 图像预处理
        image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 1. 寻找所有独立的物体轮廓
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有检测到前景，返回一个零向量
        if not contours:
            # 返回一个与正常输出维度相同的零向量
            # 这个维度需要根据下面特征的数量来确定，这里先用一个占位符
            return None 

        all_features_per_contour = []
        
        for cnt in contours:
            # 过滤掉非常小的噪声轮廓
            if cv2.contourArea(cnt) < 32:
                continue

            contour_features = []
            
            # --- 特征 A: 轮廓和形状 (Contour & Shape) ---
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # 1. 圆形度 (Circularity) - 衡量轮廓有多像圆 (破损或弯曲会改变它)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            
            # 2. 偏心率 (Eccentricity) - 衡量轮廓的拉伸程度 (前端弯曲会改变它)
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                # 先计算比率，再检查开方根内的值是否为非负
                ratio = ma / MA if MA > 0 else 0
                value_inside_sqrt = 1 - ratio**2
                eccentricity = np.sqrt(value_inside_sqrt) if value_inside_sqrt >= 0 else 0
            except cv2.error:
                eccentricity = 0

            # 3. Hu矩 (Hu Moments) - 7个具有平移、旋转、尺度不变性的形状描述符
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log-transform for scale invariance and stability
            for i in range(len(hu_moments)):
                # 加上一个极小值(epsilon)来防止log(0)产生-inf
                value = abs(hu_moments[i]) + 1e-7 
                hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(value)


            contour_features.extend([area, perimeter, circularity, eccentricity])
            contour_features.extend(hu_moments)

            # 创建一个用于提取颜色和纹理的掩码
            single_contour_mask = np.zeros(gray_image.shape, dtype="uint8")
            cv2.drawContours(single_contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # --- 特征 B: 颜色 (Color) ---
            # 4. HSV颜色直方图 (在H通道上) - 对颜色变化敏感
            h_hist = cv2.calcHist([hsv_image], [0], single_contour_mask, [16], [0, 180])
            cv2.normalize(h_hist, h_hist)
            contour_features.extend(h_hist.flatten())

            # --- 特征 C: 纹理 (Texture) ---
            # 5. LBP (局部二值模式) - 对污染、划痕等纹理变化敏感
            # 注意：LBP需要处理的区域不能太小
            x, y, w, h = cv2.boundingRect(cnt)
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_mask = single_contour_mask[y:y+h, x:x+w]
            if roi_gray.size > 0:
                P, R = 8, 1
                lbp = local_binary_pattern(roi_gray, P, R, method="uniform")
                # 只计算掩码内的LBP直方图
                (lbp_hist, _) = np.histogram(lbp[roi_mask > 0], bins=np.arange(0, P + 3), range=(0, P + 2))
                lbp_hist = lbp_hist.astype("float")
                lbp_hist /= (lbp_hist.sum() + 1e-6)
                contour_features.extend(lbp_hist)
            else: # 如果ROI为空，则填充0
                contour_features.extend(np.zeros(10)) # LBP "uniform" 有 P+2 个bin

            all_features_per_contour.append(np.array(contour_features))

        # 如果所有轮廓都被过滤掉了，返回零向量
        if not all_features_per_contour:
            return None 

        # --- 特征聚合 ---
        # 使用均值和标准差来聚合所有检测到的物体的特征，这样即使物体数量变化，特征向量长度也固定
        # 这对于检测"Missing_separator"非常重要
        features_matrix = np.array(all_features_per_contour)
        mean_features = np.mean(features_matrix, axis=0)
        std_features = np.std(features_matrix, axis=0)
        
        # 最终的特征向量
        final_feature_vector = np.concatenate([mean_features, std_features])
        # 将任何可能残余的NaN, inf, -inf值都替换为0
        final_feature_vector = np.nan_to_num(final_feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
        return final_feature_vector
    
    def extract_traditional_features_v2(self, img_rgb: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray | None:
        """
        【优化版】从图像和掩码中提取传统CV特征。
        该函数通过更优的特征选择和聚合方式，增强了对单个异常物体的敏感度，
        同时保持返回一个固定长度的特征向量。

        Args:
            img_rgb (np.ndarray): rgb格式的输入图像 (H, W, 3), uint8.
            foreground_mask (np.ndarray): 前景掩码 (H, W), uint8, 0代表背景, >0代表前景.

        Returns:
            np.ndarray | None: 包含所有聚合后特征的一维Numpy数组，如果无有效轮廓则返回None。
        """
        # 0. 图像预处理
        gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lab_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # 1. 寻找所有独立的物体轮廓
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 32]

        # 如果没有检测到前景，直接返回None或一个预定义的零向量
        if not valid_contours:
            return None 

        all_features_per_contour = []
        
        for cnt in valid_contours:
            contour_features = []
            moments = cv2.moments(cnt)
            area = moments['m00']
            
            # --- 特征 A: 形状 (Shape) - 优化选择 ---
            # 1. Hu矩 (7个) - 保留，非常稳定
            hu_moments = cv2.HuMoments(moments).flatten()
            log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
            contour_features.extend(log_hu_moments)

            # 2. 密实度 (Solidity) - 【新增】对破损/凹陷敏感
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            contour_features.append(solidity)
            
            # 3. 延展度 (Aspect Ratio) - 【新增】对弯曲/拉伸敏感
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            contour_features.append(aspect_ratio)

            # 4. 圆形度 (Circularity) - 保留
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            contour_features.append(circularity)

            # --- 特征 B: 颜色 (Color) - 优化方法 ---
            # 5. L*a*b* 颜色统计量 (6个) - 【改进】更高效、更符合感知
            single_contour_mask = np.zeros(gray_image.shape, dtype="uint8")
            cv2.drawContours(single_contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            
            for i in range(3): # L, a, b 三个通道
                channel_pixels = lab_image[:, :, i][single_contour_mask > 0]
                if channel_pixels.size > 0:
                    mean, std = cv2.meanStdDev(channel_pixels)
                    contour_features.extend([mean[0][0], std[0][0]])
                else:
                    contour_features.extend([0, 0])

            # --- 特征 C: 纹理 (Texture) - 保留 ---
            # 6. LBP 直方图 (10个)
            P, R = 8, 1
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_mask = single_contour_mask[y:y+h, x:x+w]
            
            if roi_gray.size > 0:
                lbp = local_binary_pattern(roi_gray, P, R, method="uniform")
                (lbp_hist, _) = np.histogram(lbp[roi_mask > 0], bins=np.arange(0, P + 3), range=(0, P + 2))
                lbp_hist = lbp_hist.astype("float")
                lbp_hist /= (lbp_hist.sum() + 1e-6)
                contour_features.extend(lbp_hist)
            else:
                contour_features.extend(np.zeros(P + 2))

            all_features_per_contour.append(np.array(contour_features))

        # --- 特征聚合 (Aggregation) - 关键改进 ---
        features_matrix = np.array(all_features_per_contour)
        
        # a. 计算更丰富的统计量来描述这组物体的特征分布
        mean_features = np.mean(features_matrix, axis=0)
        std_features = np.std(features_matrix, axis=0)
        median_features = np.median(features_matrix, axis=0)

        # c. 拼接所有特征形成最终的、固定长度的向量
        final_feature_vector = np.concatenate([
            # object_count, 
            mean_features, 
            std_features, 
            # min_features, 
            # max_features, 
            median_features
        ])
        
        # 替换所有可能出现的NaN或inf值，增强鲁棒性
        final_feature_vector = np.nan_to_num(final_feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
        return final_feature_vector