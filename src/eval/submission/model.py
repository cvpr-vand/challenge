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

    def generate_masks_formatted(self, raw_image_np_rgb):
        if self.model_fastsam is None:
            print("FastSAM model not loaded.")
            return []

        height, width = raw_image_np_rgb.shape[:2]
        
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
       
        # self.model_clip, _, _ = open_clip.create_model_and_transforms(
        #     model_name="ViT-L-14",
        #     pretrained="/home/dancer/LogSAD/clip_vitl14_model/open_clip_pytorch_model.bin"
        # )
        # self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')


        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        # --- 策略3：使用更小的SAM模型 ---
        SAM_MODEL_TYPE = "vit_h" #  "vit_h"、"vit_b"
        SAM_CHECKPOINT_URL_BASE = "https://dl.fbaipublicfiles.com/segment_anything/"
        SAM_CHECKPOINTS = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        checkpoint_url = SAM_CHECKPOINT_URL_BASE + SAM_CHECKPOINTS[SAM_MODEL_TYPE]
        self.model_sam = sam_model_registry[SAM_MODEL_TYPE]()
        try:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location=self.device)
            self.model_sam.load_state_dict(state_dict)
            # print(f"Successfully loaded SAM model: {SAM_MODEL_TYPE}")
        except Exception as e:
            print(f"下载或加载 SAM ({SAM_MODEL_TYPE}) 权重失败: {e}")
            print("请检查网络连接、URL或本地权重路径是否正确。")
        self.model_sam.to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(model = self.model_sam)
        # --- 策略2：调整SamAutomaticMaskGenerator参数 ---
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=self.model_sam,
        #     points_per_side=8,  # 默认 32。减少采样点数，显著加快速度。可以尝试 16, 8。
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

            # standardization
            standard_structural_score = (
                structural_score - self.stats[self.class_name]["structural_scores"]["mean"]
            ) / self.stats[self.class_name]["structural_scores"]["unbiased_std"]
            standard_instance_hungarian_match_score = (
                instance_hungarian_match_score - self.stats[self.class_name]["instance_hungarian_match_scores"]["mean"]
            ) / self.stats[self.class_name]["instance_hungarian_match_scores"]["unbiased_std"]

            pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
            pred_score = sigmoid(pred_score)

            if self.anomaly_flag:
                pred_score = 1.0
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
                topk_vals, _ = normal_map_patchcore.topk(k=3, dim=1)
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
                topk_vals, _ = normal_map_patchcore.topk(k=3, dim=1)
                normal_map_patchcore = topk_vals.mean(1).cpu().numpy()
                # normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal   
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

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
        if self.class_name == 'splicing_connectors':
            masks = self.mask_generator.generate(raw_image)
        else:
            masks = self.segmenter.generate_masks_formatted(raw_image)
        # self.predictor.set_image(raw_image)

        # # --- 策略1：降低SAM输入分辨率 ---
        # original_height, original_width = raw_image.shape[:2]
        # # 设定一个较低的目标尺寸给SAM，例如原始尺寸的一半或固定值
        # SAM_TARGET_SIZE_H = original_height // 2
        # SAM_TARGET_SIZE_W = original_width // 2

        # # 使用 cv2.resize 调整图像大小，INTER_AREA 通常用于缩小图像
        # raw_image_for_sam = cv2.resize(raw_image, (SAM_TARGET_SIZE_W, SAM_TARGET_SIZE_H), interpolation=cv2.INTER_AREA)

        # # 使用缩小后的图像生成掩码
        # # 确保 self.mask_generator 已经初始化
        # masks_low_res = self.mask_generator.generate(raw_image_for_sam)

        # # 对 SAM 生成的掩码进行处理 plot_results_only 返回一个在低分辨率下的实例分割图 (例如，每个对象一个唯一ID)
        # if masks_low_res:
        #     sorted_masks_low_res = sorted(masks_low_res, key=(lambda x: x['area']), reverse=True)
        #     # 在低分辨率下创建实例图
        #     sam_instance_map_low_res = plot_results_only(sorted_masks_low_res) # 假设 plot_results_only 返回的是 H_low x W_low 的图
        #     # 将低分辨率的实例分割图上采样回原始图像尺寸
        #     # 使用 INTER_NEAREST 来保持掩码的离散标签性质，避免模糊
        #     sam_mask = cv2.resize(sam_instance_map_low_res.astype(np.uint8), # astype很重要，resize需要数值类型
        #                         (original_width, original_height), 
        #                         interpolation=cv2.INTER_NEAREST).astype(int)
        # else:
        #     sam_mask = np.zeros((original_height, original_width), dtype=int)
        
        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy()
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1) 
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sam_mask = plot_results_only(sorted_masks).astype(int)
        
        resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        merge_sam = merge_segmentations(sam_mask, resized_mask, background_class=self.classes-1)

        resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)
        
        # filter small region for merge sam
        binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32: # 448x448 
                merge_sam[temp_mask] = self.classes - 1 # set to background

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
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            pushpins_count = num_labels - 1 # number of pushpins

            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            if self.few_shot_inited and pushpins_count != self.pushpins_count and self.anomaly_flag is False:
                self.anomaly_flag = True
                print('number of pushpins: {}, but canonical number of pushpins: {}'.format(pushpins_count, self.pushpins_count))
            
            # patch hist 
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            binary_foreground = dilate_binary.astype(np.uint8) 

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
            return {"score": score, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'splicing_connectors':
            #  object count hist for default
            # # --- 策略1：降低SAM输入分辨率 ---
            # largest_mask_high_res_bool = (sam_mask == 1)
            # sam_mask_max_area = largest_mask_high_res_bool # background
            sam_mask_max_area = sorted_masks[0]['segmentation'] # background
            binary = (sam_mask_max_area == 0).astype(np.uint8) # sam_mask_max_area is background,  background 0 foreground 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448 64
                    binary[temp_mask] = 0 # set to background
                else:
                    count += 1
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

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset

    
    
    def process(self, class_name: str, few_shot_samples: list[torch.Tensor]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device)
        scores, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(class_name, few_shot_samples)
