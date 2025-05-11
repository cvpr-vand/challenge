"""Model for submission."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


import torch
from anomalib.data import ImageBatch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize

import cv2
import random
import pickle
import numpy as np
from tabulate import tabulate
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from prompt_ensemble import encode_text_with_prompt_ensemble, encode_normal_text, encode_abnormal_text, encode_general_text, encode_obj_text
from kmeans_pytorch import kmeans, kmeans_predict


from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import open_clip_local as open_clip
# from open_clip_local.pos_embed import get_2d_sincos_pos_embed


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


def save_visualization(image_list, title_list, output_path, figsize=(20, 5)):
    """
    Save a list of images with corresponding titles to a file.

    Args:
        image_list (list): List of images to visualize.
        title_list (list): List of titles for each image.
        output_path (str): Path to save the visualization image.
        figsize (tuple): Size of the figure (width, height).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(image_list), figsize=figsize)
    
    if len(image_list) == 1:
        axes = [axes]

    for ax, temp_title, temp_image in zip(axes, title_list, image_list):
        ax.imshow(temp_image, cmap='gray' if temp_image.ndim == 2 else None)
        ax.set_title(temp_title)
        ax.margins(0, 0)
        ax.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  

class Model(nn.Module):
    """TODO: Implement your model here"""

    def __init__(self) -> None:
        super().__init__()

        setup_seed(42)

        self.img_size = 448
        # NOTE: Create your transformation pipeline (if needed).
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = v2.Compose(
            [
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )
        # 加载 CLIP 模型
        # self.model_clip, _, _ = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XYZ",img_size= 448)
        # self.tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-L-14-DataComp.XYZ")
        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',img_size= 448) #clip model
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K') #get tokenizer of clip 
        
        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024
        self.memory_size = 2048
        self.n_neighbors = 2 

        self.model_clip.eval()
        self.align_corners = True # False
        self.antialias = True # False
        self.inter_mode = 'bilinear' # bilinear/bicubic 
        
        self.cluster_feature_id = [0, 1]

        self.cluster_num_dict = {
            "breakfast_box": 3, 
            "juice_bottle": 8, 
            "splicing_connectors": 10, 
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

        # self.feat_size = 64
        # self.ori_feat_size = 32
        self.feat_size = int(self.img_size // 14) * 2
        self.ori_feat_size = int(self.img_size // 14)
        self.visualization = False

        self.pushpins_count = 15
        self.pushpins_v_channel_threshold = [20, 50]
        self.pushpins_s_channel_threshold = [170, 255]

        self.splicing_connectors_count = [2, 3, 5] # coresponding to yellow, blue, and red
        self.splicing_connectors_distance = 0
        self.splicing_connectors_cable_color_query_words_dict = [['yellow cable', 'yellow wire'], ['blue cable', 'blue wire'], ['red cable', 'red wire']]
        
        self.screw_bag_cicrle_image_shape = [652, 448]
        self.screw_bag_circle_radius = [23, 25]
        self.screw_bag_circle_count = 2

        self.juice_bottle_liquid_query_words_dict = [['red liquid', 'cherry juice'], ['yellow liquid', 'orange juice'], ['milky liquid']]
        self.juice_bottle_fruit_query_words_dict = ['cherry', ['tangerine', 'orange'], 'banana'] 

        # query words
        self.foreground_pixel_hist = 0  
        # patch query words
        self.patch_token_hist = []

        self.few_shot_inited = False

        # 加载 DINOv2 模型
        from dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        current_path = os.path.dirname(__file__)
        pkl_file = os.path.join(current_path, "memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl")
        self.stats = pickle.load(open(pkl_file, "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False #True #False


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

        print(f"======================================={class_name}=======================================")

        self.k_shot = few_shot_samples.size(0)
        self.few_shot_inited = False
        self.process(class_name, few_shot_samples)
        self.few_shot_inited = True


    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        return None


    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # TODO: Implement the forward pass of the model.
        # batch_size = image.shape[0]
        # return ImageBatch(
        #     image=image,
        #     pred_score=torch.zeros(batch_size, device=image.device),
        # )

        # force to use batch size 1
        batch_size = image.shape[0]
        if batch_size > 1:
            raise RuntimeError("out of memory")

        batch = image.clone().detach()
        # 
        self.anomaly_flag = False
        batch = self.transform(batch).to(self.device)

        # 256 -> 448
        #batch_resize = F.interpolate(batch, size=(448, 448), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        batch_resize = F.interpolate(batch, size=(self.img_size, self.img_size), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        results = self.forward_one_sample(batch_resize, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset)

        hist_score = results['hist_score']
        structural_score = results['structural_score']


        anomaly_map_structural = results['anomaly_map_structural']

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        # standardization
        standard_structural_score = (structural_score - self.stats[self.class_name]["structural_scores"]["mean"]) / self.stats[self.class_name]["structural_scores"]["unbiased_std"]
        #standard_instance_hungarian_match_score = (instance_hungarian_match_score - self.stats[self.class_name]["instance_hungarian_match_scores"]["mean"]) / self.stats[self.class_name]["instance_hungarian_match_scores"]["unbiased_std"]

        pred_score = standard_structural_score
        
        pred_score = sigmoid(pred_score)
        
        if self.anomaly_flag:
            pred_score = 1.
            self.anomaly_flag = False

        # return {"pred_score": torch.tensor(pred_score), "anomaly_map": torch.tensor(anomaly_map_structural), "hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score)}
        # return {"pred_score": torch.tensor(pred_score), "anomaly_map": torch.tensor(anomaly_map_structural), "hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score), "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}
        return ImageBatch(
            image=image,
            pred_score=torch.tensor(pred_score).to(image.device),
        )


    def set_viz(self, viz):
        self.visualization = viz


    def set_val(self, val):
        self.validation = val


    def forward_one_sample(self, batch: torch.Tensor, mem_patch_feature_clip_coreset: torch.Tensor, mem_patch_feature_dinov2_coreset: torch.Tensor):

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
             
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name)
        hist_score = results['score']

        '''calculate patchcore'''
        anomaly_maps_patchcore = []

        if self.class_name in ['pushpins', 'screw_bag']: # clip feature for patchcore
            len_feature_list = len(self.feature_list)
            for patch_feature, mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                if self.class_name in ['pushpins']:
                    normal_scores = normal_map_patchcore.topk(5, dim=1)[0].mean(dim=1) 
                    anomaly_map_patchcore = 1 - normal_scores.cpu().numpy()
                    anomaly_maps_patchcore.append(anomaly_map_patchcore)
                else:
                    normal_scores = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal
                    anomaly_maps_patchcore.append(1 - normal_scores )


        if self.class_name in ['splicing_connectors', 'breakfast_box', 'juice_bottle']: # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for patch_feature, mem_patch_feature in zip(patch_tokens_dinov2.chunk(len_feature_list, dim=-1), mem_patch_feature_dinov2_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal   
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        structural_score = np.stack(anomaly_maps_patchcore).mean(0).max()  #S_p
        anomaly_map_structural = np.stack(anomaly_maps_patchcore).mean(0).reshape(self.feat_size, self.feat_size)


        results = {'hist_score': hist_score, 'structural_score': structural_score, "anomaly_map_structural": anomaly_map_structural}

        return results


    def histogram(self, image, cluster_feature, proj_patch_token, class_name):
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

            max_label_a = int(unique_labels_a.max())
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
 


        raw_image = to_np_img(image[0]) #[448, 448, 3]
        height, width = raw_image.shape[:2]
        
        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy() 
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1)  
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation = cv2.INTER_NEAREST) 

        score = 0. # default to normal
        self.anomaly_flag = False
        
        if self.class_name == 'pushpins':
            def split_and_check_foreground(sam_mask_max_area, binary_foreground,  rows=3, cols=5):
                """
                Divide the foreground area into a specified number of rows and columns of small grids, and check whether there is a nail in each grid.
                """
                if sam_mask_max_area.dtype != bool:
                    sam_mask_max_area = sam_mask_max_area.astype(bool)
                y_coords, x_coords = np.where(sam_mask_max_area)
                if len(x_coords) == 0 or len(y_coords) == 0:
                    raise ValueError("No valid foreground pixels found in sam_mask_max_area.")
                
                min_x, max_x = x_coords.min(), x_coords.max()
                min_y, max_y = y_coords.min(), y_coords.max()

                width = max_x - min_x + 1
                cell_width = width // cols
                height = max_y - min_y + 1
                cell_height = height // rows

                grid_pixel_counts = []
                
                for row in range(rows):
                    for col in range(cols):
                        x_start = min_x + col * cell_width
                        x_end = min_x + (col + 1) * cell_width if col < cols - 1 else max_x + 1
                        y_start = min_y + row * cell_height
                        y_end = min_y + (row + 1) * cell_height if row < rows - 1 else max_y + 1

                        cell_mask = binary_foreground[y_start:y_end, x_start:x_end]
                        
                        pixel_count = np.sum(cell_mask)
                        grid_pixel_counts.append(pixel_count)

                return grid_pixel_counts
            
            # form hsv_img get details
            img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            _ , s_channel, v_channel = cv2.split(hsv_img)

            # detect the missing shield and number of pushpins
            (low_threshold , high_threshold) = self.pushpins_v_channel_threshold
            thresh_rooms = cv2.inRange(v_channel, low_threshold, high_threshold)
            num_lab_r, lab_r ,stats_r ,centroids_r = cv2.connectedComponentsWithStats(thresh_rooms, connectivity=8) #each pushpins room
            
            (low_threshold_pin , high_threshold_pin) = self.pushpins_s_channel_threshold
            thresh_pins = cv2.inRange(s_channel, low_threshold_pin, high_threshold_pin)
            num_lab_s, lab_s ,stats_s ,centroids_s = cv2.connectedComponentsWithStats(thresh_pins, connectivity=8) #each pushpins

            if not self.few_shot_inited:
                all_area_r=[]
                for lab_r in range(1, num_lab_r):
                    area_r = stats_r[lab_r, cv2.CC_STAT_AREA]
                    all_area_r.append(area_r)
                all_area_r_sorted = sorted(all_area_r)
                avg_area_r = np.mean(all_area_r_sorted[-self.pushpins_count:])

                all_area_s= []
                for lab_s in range(1, num_lab_s):
                    area_s = stats_s[lab_s, cv2.CC_STAT_AREA]
                    all_area_s.append(area_s)
                all_area_s_sorted = sorted(all_area_s)
                median_area_s = np.median(all_area_s_sorted)

            if self.few_shot_inited and self.avg_area_r_save != 0 and self.median_area_s_save != 0 and self.anomaly_flag is False:
                selected_area_r= []
                for lab_r in range(1, num_lab_r):
                    area_r = stats_r[lab_r, cv2.CC_STAT_AREA]
                    ratio1 = area_r / self.avg_area_r_save
                    if 1.7 <= ratio1 <= 5:
                        selected_area_r.append(area_r)
                if len(selected_area_r) >= 1: 
                    self.anomaly_flag = True 
                    print('missing shield in the box, having {} area is too large'.format(selected_area_r))
                
                #strcut defect
                selected_area_s= []
                for lab_s in range(1, num_lab_s):
                   area_s = stats_s[lab_s, cv2.CC_STAT_AREA]  
                   ratio2 = area_s / self.median_area_s_save
                   if 0.7 <= ratio2 <= 1.3: #87.38
                      selected_area_s.append(area_s)
                if len(selected_area_s) != self.pushpins_count: 
                   self.anomaly_flag = True
                   print('expected {} pushpins, but found {}: {}'.format(self.pushpins_count, len(selected_area_s), selected_area_s)) 
             
            gray_no_background = img_gray[img_gray != 0]
            pushpins_min_range_pixel_value = np.percentile(gray_no_background, 15) #save 15%
            thresh = cv2.inRange(img_gray, pushpins_min_range_pixel_value, 255)
            each_foreground_pins_counts = split_and_check_foreground(thresh, thresh_pins, rows=3, cols=5) 
            
            if self.few_shot_inited and self.each_foreground_pins_counts != 0 and self.anomaly_flag is False:
                for i , each_counts in enumerate(each_foreground_pins_counts):
                    ratio = each_counts / self.each_foreground_pins_counts
                    if (ratio > 1.6 or ratio < 0.4):    # color and number mismatch
                        print('the numer of pushpins room:{} has anomaly'.format(i))
                        self.anomaly_flag = True
                        break

            # patch hist 
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0]) 
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist) 


            if self.few_shot_inited: 
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            if not self.few_shot_inited:
                return {"score": score, "clip_patch_hist": clip_patch_hist,  "avg_area_r": avg_area_r, "median_area_s": median_area_s, "each_foreground_pins_counts": each_foreground_pins_counts}
            else:
                return {"score": score, "clip_patch_hist": clip_patch_hist}
            
        elif self.class_name == 'splicing_connectors':
            binary_clamps = (resized_mask == 0).astype(np.uint8) 
            binary_cable = (resized_mask == 1).astype(np.uint8) 
            binary = np.logical_or(binary_clamps, binary_cable).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448 64
                    binary[temp_mask] = 0
                else:
                    count += 1
            if count != 1 and self.anomaly_flag is False: # cable cut or no cable or no connector
                print('number of connected component in splicing_connectors: {}, but the default connected component is 1.'.format(count))
                self.anomaly_flag = True

            kernel = np.ones((5, 5), np.uint8)
            binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((23, 23), dtype=np.uint8)

            erode_binary = cv2.erode(binary_close, kernel) 
            h, w = erode_binary.shape
            distance = 0

            left, right = binary_clamps[:, :int(w/2)],  binary_clamps[:, int(w/2):]
            left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
            right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1] # foreground

            ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
            if self.few_shot_inited and (ratio > 1.2 or ratio < 0.8) and ratio != 0 and self.anomaly_flag is False: # left right asymmetry in clamp
                print('left and right connectors are not symmetry.')
                self.anomaly_flag = True

            # left and right centroids distance
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode_binary, connectivity=8)
            if num_labels - 1 == 2:
                centroids = centroids[1:] 
                x1, y1 = centroids[0] 
                x2, y2 = centroids[1]
                distance = np.sqrt((x1/w - x2/w)**2 + (y1/h - y2/h)**2) 
                if self.few_shot_inited and self.splicing_connectors_distance != 0 and distance != 0 and self.anomaly_flag is False:
                    ratio = distance / self.splicing_connectors_distance 
                    if ratio < 0.6 or ratio > 1.4:  # too short or too long centroids distance (cable) # 0.6 1.4
                        print('cable is too short or too long.')
                        self.anomaly_flag = True
            # patch hist 
            sam_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            if self.visualization:
                image_list = [raw_image,  kmeans_mask, patch_mask, binary, binary_clamps, binary_cable, erode_binary]
                title_list = ['raw image', 'kmeans mask', 'patch mask', 'binary', 'binary_clamps', 'binary_cable', 'erode binary']
                output_dir = os.path.join(os.getcwd(), f"./output1/output_{self.class_name}") 
                os.makedirs(output_dir, exist_ok=True)  
                output_path = os.path.join(output_dir, f"visualization_xxx.png")

                save_visualization(image_list, title_list, output_path)

            if not self.few_shot_inited:
                return {"score": score, "sam_patch_hist": sam_patch_hist, "distance": distance}
            else:
                return {"score": score, "sam_patch_hist": sam_patch_hist}
        
        elif self.class_name == 'screw_bag':
            # patch hist
            foreground_pixel_count = np.sum(np.bincount(kmeans_mask.reshape(-1))[:len(self.foreground_label_idx[self.class_name])])  # foreground pixel

            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                if ratio < 0.96 or ratio > 1.06: #  if ratio < 0.94 or ratio > 1.06 
                    print('foreground pixel histagram of screw bag: {}, the canonical foreground pixel histogram of screw bag in few shot: {}'.format(foreground_pixel_count, self.foreground_pixel_hist))
                    self.anomaly_flag = True

            binary_screw = np.isin(kmeans_mask, self.foreground_label_idx[self.class_name])
            patch_mask[~binary_screw] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])[:-1]
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()
            if not self.few_shot_inited:
                return {"score": score, "foreground_pixel_count": foreground_pixel_count, "clip_patch_hist": clip_patch_hist}
            else:
                return {"score": score, "clip_patch_hist": clip_patch_hist}
        
        elif self.class_name == 'breakfast_box':
            # patch hist
            sam_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0]) 
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()
            return {"score": score, "sam_patch_hist": sam_patch_hist}
        
        elif self.class_name == 'juice_bottle': 

            sam_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:  
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T) 
                score = 1 - patch_hist_similarity.max()
            return {"score": score, "sam_patch_hist": sam_patch_hist}

        return {"score": score}


    def process_k_shot(self, class_name, few_shot_samples):
        few_shot_samples = F.interpolate(few_shot_samples, size=(self.img_size, self.img_size), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)

        with torch.no_grad(): 
            print(f"Input shape: {few_shot_samples.shape}")
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(few_shot_samples, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]  #delet cls token
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (bs*1024, 1024x4) 
            patch_tokens_clip = patch_tokens_clip.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2) #(4,4096,32,32)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners) # (4,4096,64,64)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list)) # (bsx64x64, 1024x4) (16384，4096)
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1) 

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(few_shot_samples, out_layer_list=self.feature_list_dinov2)  # 4 x [bs, 32x32, 1024]
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (bs, 1024, 1024x4)  torch.Size([4, 1024, 4096])
            patch_tokens_dinov2 = patch_tokens_dinov2.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (bsx64x64, 1024x4) (16384，4096)

        cluster_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            cluster_features = temp_feat if cluster_features is None else torch.cat((cluster_features, temp_feat), 1) #(bsx64x64, 1024x2)
        if self.feat_size != self.ori_feat_size:
            cluster_features = cluster_features.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2) #torch.Size([4, 2048, 32, 32])
            cluster_features = F.interpolate(cluster_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            cluster_features = cluster_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id)) #(16384，2048)
        cluster_features = F.normalize(cluster_features, p=2, dim=-1)

        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2) #torch.Size([4, 768, 32, 32])
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim) #(16384，768) 
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        num_clusters = self.cluster_num_dict[class_name] 
        _, self.cluster_centers = kmeans(X=cluster_features, num_clusters=num_clusters, device=self.device) 

        self.query_obj = encode_obj_text(self.model_clip, self.query_words_dict[class_name], self.tokenizer, self.device) #[2,768]
        self.patch_query_obj = encode_obj_text(self.model_clip, self.patch_query_words_dict[class_name], self.tokenizer, self.device) #[2,768]
        self.classes = self.query_obj.shape[0] #2

        scores = []
        foreground_pixel_hist = []
        splicing_connectors_distance = []
        patch_token_hist = []
        avg_area_r_save = []
        median_area_s_save = []
        each_foreground_pins_counts = []


        for image, cluster_feature, proj_patch_token in zip(few_shot_samples.chunk(self.k_shot), cluster_features.chunk(self.k_shot), proj_patch_tokens.chunk(self.k_shot)):        
            self.anomaly_flag = False
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name)
            if self.class_name == 'pushpins':
                patch_token_hist.append(results["clip_patch_hist"])
                avg_area_r_save.append(results['avg_area_r'])
                median_area_s_save.append(results['median_area_s'])
                each_foreground_pins_counts.append(results['each_foreground_pins_counts'])

            elif self.class_name == 'splicing_connectors':
                splicing_connectors_distance.append(results["distance"])
                patch_token_hist.append(results["sam_patch_hist"])

            elif self.class_name == 'screw_bag':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                patch_token_hist.append(results["clip_patch_hist"])

            elif self.class_name == 'breakfast_box':
                patch_token_hist.append(results["sam_patch_hist"])

            elif self.class_name == 'juice_bottle':
                patch_token_hist.append(results["sam_patch_hist"])

            scores.append(results["score"])

        # pushpins
        if len(avg_area_r_save) != 0:
            self.avg_area_r_save = np.mean(avg_area_r_save)
        if len(median_area_s_save) != 0:
            self.median_area_s_save = np.mean(median_area_s_save)
        if len(each_foreground_pins_counts) != 0:
            flattened_foreground_pixel_hist = np.concatenate(each_foreground_pins_counts)
            self.each_foreground_pins_counts = np.mean(flattened_foreground_pixel_hist)


        if len(foreground_pixel_hist) != 0:
            self.foreground_pixel_hist = np.mean(foreground_pixel_hist)
        if len(splicing_connectors_distance) != 0:
            self.splicing_connectors_distance = np.mean(splicing_connectors_distance)
        if len(patch_token_hist) != 0: # patch hist
            self.patch_token_hist = np.stack(patch_token_hist)

        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset


    def process(self, class_name: str, few_shot_samples: list[torch.Tensor]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device) # [k, 3, 448, 448] 
        scores, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(class_name, few_shot_samples)
