"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn
from torchvision.transforms import v2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use('Agg')

import pickle
from scipy.stats import norm

#import open_clip_local as open_clip
import eval.submission.open_clip_local as open_clip
import cv2
import numpy as np
import torch.nn.functional as F
from kmeans_pytorch import kmeans, kmeans_predict
from eval.submission.utils.prompt_ensemble import encode_text_with_prompt_ensemble, encode_normal_text, encode_abnormal_text, encode_general_text, encode_obj_text
from scipy.optimize import linear_sum_assignment
import time
import h5py
import hashlib
import os
import json
from typing import List, Dict, Any
import random
from timm import create_model

import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
cur_path = os.path.join(dir_path, "gdino_sam2")
#sys.path.append('/home/user/actions-runner/_work/challenge/challenge/src/eval/submission/gdino_sam2')
sys.path.append(cur_path)
print("cur_path:", cur_path)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from eval.submission.gdino_sam2.interface import GSAM2Predictor
import requests

def to_np_img(m):
    m = m.permute(1, 2, 0).cpu().numpy()
    mean = np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    std = np.array([[[0.26862954, 0.26130258, 0.27577711]]])
    m  = m * std + mean
    return np.clip((m * 255.), 0, 255).astype(np.uint8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_weights_with_requests(url, save_path):
    """
    使用 requests 下载权重文件

    参数:
        url: 下载链接
        save_path: 保存路径（包含文件名）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"权重文件已下载到: {save_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        
class Model(nn.Module):
    """TODO: Implement your model here"""

    def __init__(self) -> None:
        super().__init__()

        # NOTE: Create your transformation pipeline (if needed).
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = v2.Compose(
            [
                v2.Resize((448, 448)),
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )

        # NOTE: Create your model.


        self.model_clip, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

        #local_model_path = '/workspace/MyDeptEDS/srj/2025_competition/LogSAD-master/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin'
        #self.model_clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=local_model_path)
        #self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        #self.model_sam = sam_model_registry["vit_h"](checkpoint="./submission/checkpoint/sam_vit_h_4b8939.pth").to(self.device)
        # self.model_sam = sam_model_registry["vit_b"](checkpoint = "./submission/checkpoint/sam_vit_b_01ec64.pth").to(self.device)

        import os
        from urllib.request import urlretrieve
        
        SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        CHECKPOINT_DIR = "./checkpoint"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")

        if not os.path.exists(MODEL_PATH):
            print("Downloading SAM model...")
            urlretrieve(SAM_MODEL_URL, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        self.model_sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)

        #self.model_sam = sam_model_registry["vit_h"](checkpoint="./submission/checkpoint/sam_vit_h_4b8939.pth").to(self.device)
        self.mask_generator_sam = SamAutomaticMaskGenerator(model=self.model_sam)
        
        #sam2_checkpoint = "./submission/checkpoint/sam2.1_hiera_large.pt"
        #model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        #sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("current_dir:", current_dir)
        SAM2_CHECKPOINT = os.path.join(current_dir, "checkpoint/sam2.1_hiera_large.pt")
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if not os.path.exists(SAM2_CHECKPOINT):
            download_weights_with_requests(
                url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                save_path=SAM2_CHECKPOINT
            )
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)

        self.mask_generator_sam2 = SAM2AutomaticMaskGenerator(sam2)
        sam2_predictor = SAM2ImagePredictor(sam2)

        self.img_size = 448

        self.memory_size = 2048
        self.n_neighbors = 2

        self.model_clip.eval()
        self.test_args = None
        self.align_corners = True  # False
        self.antialias = True  # False
        self.inter_mode = 'bilinear'  # bilinear/bicubic

        self.cluster_feature_id = [0, 1]

        self.cluster_num_dict = {
            "breakfast_box": 3,  # unused
            "juice_bottle": 8,  # unused
            "splicing_connectors": 10,  # unused
            "pushpins": 10,
            "screw_bag": 10,
        }
        self.query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box',
                              'black background'],
            "juice_bottle": ['bottle', ['black background', 'background']],
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['screw'], 'plastic bag', 'background'],
            "splicing_connectors": [['splicing connector', 'splice connector', ], ['cable', 'wire'], ['grid']],
        }
        self.foreground_label_idx = {  # for query_words_dict
            "breakfast_box": [0, 1, 2, 3, 4, 5],
            "juice_bottle": [0],
            "pushpins": [0],
            "screw_bag": [0],
            "splicing_connectors": [0, 1]
        }

        self.patch_query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box',
                              'black background'],
            "juice_bottle": [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'],
                             ['black background', 'background']],
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['hex screw', 'hexagon bolt'], ['hex nut', 'hexagon nut'], ['ring washer', 'ring gasket'],
                          ['plastic bag', 'background']],
            "splicing_connectors": [['splicing connector', 'splice connector', ], ['cable', 'wire'], ['grid']],
        }

        self.query_threshold_dict = {
            "breakfast_box": [0., 0., 0., 0., 0., 0., 0.],  # unused
            "juice_bottle": [0., 0., 0.],  # unused
            "splicing_connectors": [0.15, 0.15, 0.15, 0., 0.],  # unused
            "pushpins": [0.2, 0., 0., 0.],
            "screw_bag": [0., 0., 0., ],
        }
        self.feat_size = 64
        self.ori_feat_size = 32
        self.visualization = False
        self.pushpins_count = 15

        self.splicing_connectors_count = [2, 3, 5]  # coresponding to yellow, blue, and red
        self.splicing_connectors_distance = 0
        self.splicing_connectors_cable_color_query_words_dict = [['yellow cable', 'yellow wire'],
                                                                 ['blue cable', 'blue wire'], ['red cable', 'red wire']]

        self.juice_bottle_liquid_query_words_dict = [['red liquid', 'cherry juice'], ['yellow liquid', 'orange juice'],
                                                     ['milky liquid']]
        self.juice_bottle_fruit_query_words_dict = ['cherry', ['tangerine', 'orange'], 'banana']

        # query words
        self.foreground_pixel_hist = 0
        # patch query words
        self.patch_token_hist = []

        self.few_shot_inited = False

        from eval.submission.dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024
        #self.stats = pickle.load(open("/workspace/MyDeptEDS/srj/2025_competition/challenge-main/LogSAD/src/eval/submission/memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl", "rb"))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        val_path = os.path.join(current_dir, "memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl")
        self.stats = pickle.load(open(val_path, "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False  # True #False

        ###yangye
        self.image_size = 448
        self.out_layers = [5, 11, 17, 23]
        self.embedding_dim = 1024
        self.perform_grid_size = 64
        self.transform_dino = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
        )
        self.objs_dict = {
            # 5 大类别
            "breakfast_box": [["white box"], ["orange",  "orange peach"], ["peach",], ["oatmeal"], ["banana chips", "almonds", "banana chips almonds "]],
            # 5 大类别
            "juice_bottle": [["glass bottle"], ["cherry", "orange", "banana"], ["label", "tag", "label tag"]],
            # 两个类别
            "pushpins": [["tools"]],
            # 3个类别
            "screw_bag": [["plastic bag"], ["metal circle", "circle"], ["long bolts", "bolt"]],
            # 2个类别
            "splicing_connectors": [["connector"]],
        }
        self.foreground_num = {  # for query_words_dict
            "breakfast_box": 5,
            "juice_bottle": 3,
            "pushpins": 2,
            "screw_bag": 3,
            "splicing_connectors":2
        }
        #eva02_pretrained_cfg = {"file": "submission/eva02_large_patch14_clip_336.merged2b_s6b_b61k/open_clip_model.safetensors"}
        self.model_eva02 = create_model('eva02_large_patch14_clip_336', pretrained=True, num_classes=768, img_size=self.image_size).to(self.device)
        #self.model_eva02 = create_model('eva02_large_patch14_clip_336', pretrained=True,pretrained_cfg=eva02_pretrained_cfg, num_classes=768,img_size=self.image_size).to(self.device)
        self.model_eva02.eval()
        # dinov2_pretrained_cfg = {"file": "models/vit_large_patch14_dinov2.lvd142m/model.safetensors"}
        #dinov2_pretrained_cfg = {"file": "submission/vit_large_patch14_dinov2.lvd142m/model.safetensors"}
        self.model_dinov2_yy = create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=self.image_size).to(self.device)
        #self.model_dinov2_yy = create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True,pretrained_cfg=dinov2_pretrained_cfg, num_classes=0,img_size=self.image_size).to(self.device)
        self.model_dinov2_yy.eval()
        self.gdino_init = False
        self.vis = False
        self.class_hists_train = None
        self.class_color_train = None
        self.mask_generator = GSAM2Predictor(sam2_predictor)
        self.gdino_init = True



    def set_viz(self, viz):
        self.visualization = viz

    def set_val(self, val):
        self.validation = val

    def histogram(self, image, cluster_feature, proj_patch_token, class_name, path):
        if self.class_name == 'pushpins':
            pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device) 
            kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)    # default to background

            raw_image = to_np_img(image[0]) #[448, 448, 3]
            height, width = raw_image.shape[:2]

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

            kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy() 
            kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
            patch_mask = patch_similarity.argmax(-1)  
            patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation = cv2.INTER_NEAREST) 

            score = 0. # default to normal
            self.anomaly_flag = False

            hsv_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HSV)
            _ , s_channel, v_channel = cv2.split(hsv_img)

            # split grid
            img_grid = cv2.inRange(v_channel, 20, 50)
            num_lab_g, _ ,stats_g , _ = cv2.connectedComponentsWithStats(img_grid, connectivity=8) #each pushpins room

            # split pin
            img_pin = cv2.inRange(s_channel, 170, 255)
            num_lab_p, _ ,stats_p , _ = cv2.connectedComponentsWithStats(img_pin, connectivity=8) #each pushpins
           
            # cal area of grid、pin
            if not self.few_shot_inited:
                grid_areas_tmp = [stats_g[i, cv2.CC_STAT_AREA] for i in range(1, num_lab_g)]
                grid_areas_tmp.sort()
                grid_area = np.mean(grid_areas_tmp[-self.pushpins_count : ])

                pin_areas_tmp = [stats_p[j, cv2.CC_STAT_AREA] for j in range(1, num_lab_p)]
                pin_area = np.median(pin_areas_tmp)

            # 判断grid是否异常(area、pin数量)
            if self.few_shot_inited and self.grid_area != 0 and self.pin_area != 0 and self.anomaly_flag is False:
                error_grids = [i for i in range(1, num_lab_g) if 1.7<=stats_g[i, cv2.CC_STAT_AREA]/self.grid_area<=5]
                if len(error_grids): 
                    self.anomaly_flag = True 
                    print('grid {} area anomaly.'.format(error_grids))

                right_grids = [i for i in range(1, num_lab_p) if 0.7<=stats_p[i, cv2.CC_STAT_AREA]/self.pin_area<=1.3]
                if len(right_grids) != self.pushpins_count: 
                   self.anomaly_flag = True
                   print('number of pushpins: {}, but canonical number of pushpins: {}'.format(len(right_grids), self.pushpins_count))

            img_gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
            foreground_thresh = np.percentile(img_gray[img_gray != 0], 15)
            img_foreground = cv2.inRange(img_gray, foreground_thresh, 255)
            
            # pinpixel count per grid
            nonzero_pixels = cv2.findNonZero(img_foreground)
            xs = [pt[0][0] for pt in nonzero_pixels]
            ys = [pt[0][1] for pt in nonzero_pixels]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            grid_w = (max_x - min_x + 1) / 5
            grid_h = (max_y - min_y + 1) / 3

            each_grid_pins_counts = []
            for row in range(3):
                for col in range(5):
                    start_x = min_x + int(col * grid_w)
                    end_x = min_x + int((col+1) * grid_w)
                    start_y = min_y + int(row * grid_h)
                    end_y = min_y + int((row+1) * grid_h)

                    cell_mask = img_pin[start_y:end_y, start_x:end_x]

                    pinpixel_count = np.sum(cell_mask)
                    each_grid_pins_counts.append(pinpixel_count)

            if self.few_shot_inited and self.each_grid_pins_counts != 0 and self.anomaly_flag is False:
                for i , pinpixel_counts in enumerate(each_grid_pins_counts):
                    ratio = pinpixel_counts / self.each_grid_pins_counts
                    if not (0.4 <= ratio <= 1.6):
                        print('grid {}: pushpin count ≠ 1'.format(i))
                        self.anomaly_flag = True
                        break
                    
            # patch hist
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            if not self.few_shot_inited:
                return {"score": score, "clip_patch_hist": clip_patch_hist,  "grid_area": grid_area, "pin_area": pin_area, "each_grid_pins_counts": each_grid_pins_counts}
            else:
                return {"score": score, "clip_patch_hist": clip_patch_hist}

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
                    label_map[label_a] = background_class  # default background

            merged_a = label_map[a]
            return merged_a

        pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device)
        kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)  # default to background

        
        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        time1 = time.time()
        # masks = self.mask_generator.generate(raw_image)
        # 维护数据库
        self.db = TensorDictDB(self.class_name)
        result = self.db.get(raw_image)
        if result:
            masks = result
        else:
            # if self.class_name == 'juice_bottle':
            if self.class_name in ['juice_bottle', 'splicing_connectors']:
                masks = self.mask_generator_sam2.generate(raw_image)
            else:
                masks = self.mask_generator_sam.generate(raw_image)
            self.db.add(raw_image, masks)
        time2 = time.time()
        print("sam time:", time2 - time1)
        

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
                if temp_prob > self.query_threshold_dict[class_name][temp_label]:  # threshold for each class
                    kmeans_mask[mask] = temp_label


        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy()
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1)
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sam_mask = plot_results_only(sorted_masks).astype(np.int32)

        resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        merge_sam = merge_segmentations(sam_mask, resized_mask, background_class=self.classes - 1)

        resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask,
                                              background_class=self.patch_query_obj.shape[0] - 1)

        # filter small region for merge sam
        binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(
            np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32:  # 448x448
                merge_sam[temp_mask] = self.classes - 1  # set to background

        # filter small region for patch merge sam
        binary = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32:  # 448x448
                patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0] - 1  # set to background

        score = 0.  # default to normal
        self.anomaly_flag = False
        instance_masks = []
        
        if self.class_name == 'splicing_connectors':
            ######
            # 交点位置判断
            def cal_dist(binary):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
                for i in range(1, num_labels):
                    temp_mask = labels == i
                    if np.sum(temp_mask) <= 64: # 448x448 64
                        binary[temp_mask] = 0 # set to background (del 小连通域)

                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)) # 15, 15
                rect_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                # 使用连通域分析统计矩形个数(连通域数量含背景)
                num_labels_rects, labels_rects, stats_rects, _ = cv2.connectedComponentsWithStats(rect_mask, connectivity=8) # 仅保留两端
                only_line = cv2.subtract(binary, rect_mask)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                only_line = cv2.morphologyEx(only_line, cv2.MORPH_OPEN, kernel_open)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(only_line, connectivity=8)
                min_area = 1000  # 面积阈值
                final_line = np.zeros_like(only_line)
                for i in range(1, num_labels):  # 跳过背景
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        final_line[labels == i] = 255
                        
                num_labels_line, labels_line, stats_line, _ = cv2.connectedComponentsWithStats(final_line, connectivity=8)
                # 4. 找到直线的两个端点（即交点）
                nonzero_cols = np.where(np.any(final_line, axis=0))[0]  # 所有包含非零像素的列
                if len(nonzero_cols) > 0 and num_labels_rects == 3 and num_labels_line == 2:
                    try:
                        left_x = nonzero_cols[0]    # 最左点 x 坐标
                        right_x = nonzero_cols[-1] # 最右点 x 坐标
                        # 找到对应 y 坐标（取该列中间位置）
                        left_y = np.mean(np.where(final_line[:, left_x] > 0)[0])
                        right_y = np.mean(np.where(final_line[:, right_x] > 0)[0])
                        left_box = stats_rects[1,:-1]
                        right_box = stats_rects[2,:-1]
                        if left_box[0] > right_box[0]:
                            left_box, right_box = right_box, left_box
                        
                        
                        left_box_ymin = left_box[1]
                        left_box_ymax = left_box[1] + left_box[3]
                        left_box_ycenter = (left_box_ymin + left_box_ymax) / 2
                        dist1 = left_y - left_box_ycenter
                        # left_top = abs(left_y - left_box_ymin)
                        # left_bottom = abs(left_box_ymax - left_y)
                        # dist1 = abs(left_top - left_bottom)
                        
                        right_box_ymin = right_box[1]
                        right_box_ymax = right_box[1] + right_box[3]
                        right_box_ycenter = (right_box_ymin + right_box_ymax) / 2
                        dist2 = right_y - right_box_ycenter
                        # right_top = abs(right_y - right_box_ymin)
                        # right_bottom = abs(right_box_ymax - right_y)
                        # dist2 = abs(right_top - right_bottom)
                        print(dist1, dist2, dist2-dist1)

                        return abs(dist1-dist2)
                    except:
                        return 0
                else:
                    return 0
            '''    
            def cal_component_num(binary): # 计算组件数量
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 15, 15
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8) # 

                component_count = 0 # 良品为1
                for i in range(1, num_labels): # 第0个连通域为背景
                    temp_mask = labels == i # 每个连通域mask
                    if np.sum(temp_mask) <= 64:  # 448x448 64
                        binary[temp_mask] = 0  # set to background（将小面积连通域置为背景）
                    else:
                        component_count += 1

                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)) # 15, 15
                rect_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                # 使用连通域分析统计矩形个数(连通域数量含背景)
                num_labels_rects, labels_rects, stats_rects, _ = cv2.connectedComponentsWithStats(rect_mask, connectivity=8) # 仅保留两端
                only_line = cv2.subtract(binary, rect_mask)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                only_line = cv2.morphologyEx(only_line, cv2.MORPH_OPEN, kernel_open)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(only_line, connectivity=8)
                min_area = 1000  # 面积阈值
                final_line = np.zeros_like(only_line)
                for i in range(1, num_labels):  # 跳过背景
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        final_line[labels == i] = 255
                num_labels_line, labels_line, stats_line, _ = cv2.connectedComponentsWithStats(final_line, connectivity=8)
                return component_count, num_labels_rects-1, num_labels_line-1 # mask中连通域数量、两端连接器数量、线缆数量（良品对应三项数量：1，2，1）('-1'因为第0个连通域为背景)self.class_name == 'sp
            '''
            def count_red_pixels_rgb(img_bgr):
                b, g, r = cv2.split(img_bgr)
                
                # 生成红色像素的布尔掩膜 (r > g 且 r > b)
                red_mask = (r > g+125) & (r > b+125)
                
                # 统计红色像素数量
                red_pixels = np.sum(red_mask)
                return red_pixels

            def apply_binary_mask(binary, rgb_img):
                if len(binary.shape) > 2:
                    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
                
                mask = cv2.merge([binary, binary, binary])
                
                rgb_img2 = cv2.bitwise_and(rgb_img, mask)
                
                return rgb_img2
            def count_line_component(rgb_line_img, red_mask):
                red_only = rgb_line_img.copy()
                red_only[~red_mask] = 0  # ~表示逻辑非
                red_only = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
                _, red_only = cv2.threshold(red_only, 50, 255, cv2.THRESH_BINARY)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 15, 15
                red_only = cv2.morphologyEx(red_only, cv2.MORPH_CLOSE, kernel_open)
                # red_only = cv2.morphologyEx(red_only, cv2.MORPH_OPEN, kernel_open)
                        
                # 连通域分析
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_only, connectivity=8)
                # 创建空白掩膜用于保留有效区域
                filtered_mask = np.zeros_like(red_only)

                # 遍历连通域（跳过背景标签0）
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]  # 获取当前连通域面积
                    if area >= 40:  # 保留面积≥40的连通域
                        filtered_mask[labels == i] = 255  # 将该连通域设为白色
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
                return num_labels
            
                
            def cal_component_num(binary, raw_image): # 计算组件数量
                binary = np.where(binary > 0, 255, 0).astype(np.uint8)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 15, 15
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8) # 

                component_count = 0 # 良品为1
                for i in range(1, num_labels): # 第0个连通域为背景
                    temp_mask = labels == i # 每个连通域mask
                    if np.sum(temp_mask) <= 64:  # 448x448 64
                        binary[temp_mask] = 0  # set to background（将小面积连通域置为背景）
                    else:
                        component_count += 1

                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)) # 15, 15
                rect_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open) # 连接器
                # 使用连通域分析统计矩形连接器个数(连通域数量含背景)
                num_labels_rects, labels_rects, stats_rects, _ = cv2.connectedComponentsWithStats(rect_mask, connectivity=8) # 仅保留两端
                # rgb_img = cv2.imread('structural_anomalies/' + item.split('_')[2] + '.png')
                # rgb_img = cv2.imread('good/' + item.split('_')[1] + '.png')
                # rgb_img = cv2.resize(rgb_img, (448, 448))
                rgb_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

                white_pixel_nums = []
                red_pixel_nums = []
                for i in range(1, num_labels_rects): # 跳过背景
                    mask = (labels_rects == i).astype(np.uint8) * 255
                    white_pixel_num = np.count_nonzero(mask)
                    white_pixel_nums.append(white_pixel_num)
                        
                    rgb_img2 = apply_binary_mask(mask, rgb_img)
                    red_pixel_num = count_red_pixels_rgb(rgb_img2)
                    red_pixel_nums.append(red_pixel_num)
                
                only_line = cv2.subtract(binary, rect_mask)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                only_line = cv2.morphologyEx(only_line, cv2.MORPH_OPEN, kernel_open)
                
                ###
                # line
                rgb_line_img = apply_binary_mask(only_line, rgb_img)
                # 判读line颜色
                b, g, r = cv2.split(rgb_line_img)
                red_mask = (r > g + 125) & (r > b + 125)
                yellow_mask = (r > 125) & (g > 125) & (b < 100)
                blue_mask = (b > g + 25) & (b > r + 25)
                red_pixels = np.sum(red_mask)
                yellow_pixels = np.sum(yellow_mask)
                blue_pixels = np.sum(blue_mask)
                color_counts = {'red': red_pixels, 'yellow': yellow_pixels, 'blue': blue_pixels}
                dominant_color = max(color_counts, key=color_counts.get)
                
                if dominant_color == 'red':
                    num_line_component = count_line_component(rgb_line_img, red_mask)
                elif dominant_color == 'yellow':
                    num_line_component = count_line_component(rgb_line_img, yellow_mask)
                else:
                    num_line_component = count_line_component(rgb_line_img, blue_mask)
                ###
                
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(only_line, connectivity=8)
                min_area = 1000  # 面积阈值
                final_line = np.zeros_like(only_line)
                for i in range(1, num_labels):  # 跳过背景
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        final_line[labels == i] = 255
                num_labels_line, labels_line, stats_line, _ = cv2.connectedComponentsWithStats(final_line, connectivity=8)

                ratio = 0
                if len(white_pixel_nums) == 2 and (len(red_pixel_nums) == 2):
                    # connect_area_ratio = max(white_pixel_nums) / min(white_pixel_nums)
                    # red_area_ratio = max(red_pixel_nums) / min(red_pixel_nums)
                    red_count1 = red_pixel_nums[0]
                    connect_count1 = white_pixel_nums[0]
                    red_count2 = red_pixel_nums[1]
                    connect_count2 = white_pixel_nums[1]
                    ratio = max((connect_count1-red_count1)/connect_count1, (connect_count2-red_count2)/connect_count2)
                return ratio, component_count, num_labels_rects-1, num_labels_line-1, num_line_component-1 # mask中连通域数量、两端连接器数量、线缆数量（良品对应三项数量：1，2，1）('-1'因为第0个连通域为背景)

            sam_mask_max_area = sorted_masks[0]['segmentation']  # background
            binary = (sam_mask_max_area == 0).astype(
                np.uint8)  # sam_mask_max_area is background,  background 0 foreground 1
            dist = cal_dist(binary)
            if dist > 30:
                self.anomaly_flag = True
            ######

            #  object count hist for default
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            # component_count, connector_count, line_count = cal_component_num(binary)
            ratio, component_count, connector_count, line_count, num_line_component = cal_component_num(binary, raw_image)
            if (ratio > 0.53):
                self.anomaly_flag = True
            if num_line_component != 1:
                self.anomaly_flag = True
            
            if (component_count != 1 or connector_count != 2 or line_count != 1) and self.anomaly_flag is False:
                self.anomaly_flag = True

            
            merge_sam[~(binary.astype(np.bool_))] = self.query_obj.shape[0] - 1  # remove noise
            patch_merge_sam[~(binary.astype(np.bool_))] = self.patch_query_obj.shape[0] - 1  # remove patch noise

            # erode the cable and divide into left and right parts
            kernel = np.ones((23, 23), dtype=np.uint8)
            erode_binary = cv2.erode(binary, kernel)
            h, w = erode_binary.shape
            distance = 0

            left, right = erode_binary[:, :int(w / 2)], erode_binary[:, int(w / 2):]
            left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
            right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1]  # foreground
            ###
            hist_r_l = np.bincount((raw_image[:,:224,::-1]*(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)))[:,:,2].ravel(), minlength=256)[10:]
            hist_r_r = np.bincount((raw_image[:,224:,::-1]*(cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)))[:,:,2].ravel(), minlength=256)[10:]
            norm_hist_r_l = hist_r_l / np.linalg.norm(hist_r_l)
            norm_hist_r_r = hist_r_r / np.linalg.norm(hist_r_r)
            sim_r = norm_hist_r_l @ norm_hist_r_r.T
            if sim_r < 0.8:
                print('color.')
                self.anomaly_flag = True
            ###

            binary_cable = (patch_merge_sam == 1).astype(np.uint8)

            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_cable = cv2.erode(binary_cable, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cable, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:  # 448x448
                    binary_cable[temp_mask] = 0  # set to background

            binary_cable = cv2.resize(binary_cable, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)

            binary_clamps = (patch_merge_sam == 0).astype(np.uint8)

            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_clamps = cv2.erode(binary_clamps, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clamps, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:  # 448x448
                    binary_clamps[temp_mask] = 0  # set to background
                else:
                    instance_mask = temp_mask.astype(np.uint8)
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),
                                               interpolation=cv2.INTER_NEAREST)
                    if instance_mask.any():
                        instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

            binary_clamps = cv2.resize(binary_clamps, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)

            binary_connector = cv2.resize(binary, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)

            query_cable_color = encode_obj_text(self.model_clip, self.splicing_connectors_cable_color_query_words_dict,
                                                self.tokenizer, self.device)
            cable_feature = proj_patch_token[binary_cable.astype(np.bool_).reshape(-1), :].mean(0, keepdim=True)
            idx_color = (cable_feature @ query_cable_color.T).argmax(-1).squeeze(0).item()
            foreground_pixel_count = np.sum(erode_binary) / self.splicing_connectors_count[idx_color]

            # slice_cable = binary[:, int(w / 2) - 1: int(w / 2) + 1]
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
            # cable_count = num_labels - 1
            # if cable_count != 1 and self.anomaly_flag is False:  # two cables
            #     print('number of cable count in splicing_connectors: {}, but the default cable count is 1.'.format(
            #         cable_count))
            #     self.anomaly_flag = True

            # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:  # color and number mismatch
                    print(
                        'cable color and number of clamps mismatch, cable color idx: {} (0: yellow 2-clamp, 1: blue 3-clamp, 2: red 5-clamp), foreground_pixel_count :{}, canonical foreground_pixel_hist: {}.'.format(
                            idx_color, foreground_pixel_count, self.foreground_pixel_hist))
                    self.anomaly_flag = True

            # left right hist for symmetry
            ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
            if self.few_shot_inited and (
                    ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:  # left right asymmetry in clamp
                print('left and right connectors are not symmetry.')
                self.anomaly_flag = True

            # left and right centroids distance
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode_binary, connectivity=8)
            if num_labels - 1 == 2:
                centroids = centroids[1:]
                x1, y1 = centroids[0]
                x2, y2 = centroids[1]
                distance = np.sqrt((x1 / w - x2 / w) ** 2 + (y1 / h - y2 / h) ** 2)
                if self.few_shot_inited and self.splicing_connectors_distance != 0 and self.anomaly_flag is False:
                    ratio = distance / self.splicing_connectors_distance
                    if ratio < 0.6 or ratio > 1.4:  # too short or too long centroids distance (cable) # 0.6 1.4
                        print('cable is too short or too long.')
                        self.anomaly_flag = True

            # patch hist
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[
                0])  # [:-1]  # ignore background (grid) for statistic
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # todo    mismatch cable link
            binary_foreground = binary.astype(np.uint8)  # only 1 instance, so additionally seperate cable and clamps
            if binary_connector.any():
                instance_masks.append(binary_connector.astype(np.bool_).reshape(-1))
            if binary_clamps.any():
                instance_masks.append(binary_clamps.astype(np.bool_).reshape(-1))
            if binary_cable.any():
                instance_masks.append(binary_cable.astype(np.bool_).reshape(-1))

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, binary_connector, merge_sam,
                              patch_merge_sam, erode_binary, binary_cable, binary_clamps]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'binary_connector',
                              'merge sam', 'patch merge sam', 'erode binary', 'binary_cable', 'binary_clamps']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "distance": distance,
                    "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'screw_bag':
            # pixel hist of kmeans mask
            foreground_pixel_count = np.sum(np.bincount(kmeans_mask.reshape(-1))[
                                            :len(self.foreground_label_idx[self.class_name])])  # foreground pixel
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                # todo: optimize
                if ratio < 0.94 or ratio > 1.06:
                    print(
                        'foreground pixel histagram of screw bag: {}, the canonical foreground pixel histogram of screw bag in few shot: {}'.format(
                            foreground_pixel_count, self.foreground_pixel_hist))
                    self.anomaly_flag = True

            # patch hist
            binary_screw = np.isin(kmeans_mask, self.foreground_label_idx[self.class_name])
            patch_mask[~binary_screw] = self.patch_query_obj.shape[0] - 1  # remove patch noise
            resized_binary_screw = cv2.resize(binary_screw.astype(np.uint8),
                                              (patch_merge_sam.shape[1], patch_merge_sam.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            patch_merge_sam[~(resized_binary_screw.astype(np.bool_))] = self.patch_query_obj.shape[
                                                                           0] - 1  # remove patch noise

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
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),
                                           interpolation=cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam,
                              binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask',
                              'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

            return {"score": score, "foreground_pixel_count": foreground_pixel_count,
                    "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}

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
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),
                                           interpolation=cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam,
                              binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask',
                              'patch merge sam', 'binary_foreground']
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
            resized_patch_merge_sam = cv2.resize(patch_merge_sam, (self.feat_size, self.feat_size),
                                                 interpolation=cv2.INTER_NEAREST)
            binary_liquid = (resized_patch_merge_sam == 1)
            binary_fruit = (resized_patch_merge_sam == 2)

            query_liquid = encode_obj_text(self.model_clip, self.juice_bottle_liquid_query_words_dict, self.tokenizer,
                                           self.device)
            query_fruit = encode_obj_text(self.model_clip, self.juice_bottle_fruit_query_words_dict, self.tokenizer,
                                          self.device)

            liquid_feature = proj_patch_token[binary_liquid.reshape(-1), :].mean(0, keepdim=True)
            liquid_idx = (liquid_feature @ query_liquid.T).argmax(-1).squeeze(0).item()

            fruit_feature = proj_patch_token[binary_fruit.reshape(-1), :].mean(0, keepdim=True)
            fruit_idx = (fruit_feature @ query_fruit.T).argmax(-1).squeeze(0).item()

            if (liquid_idx != fruit_idx) and self.anomaly_flag is False:
                print('liquid: {}, but fruit: {}.'.format(self.juice_bottle_liquid_query_words_dict[liquid_idx],
                                                          self.juice_bottle_fruit_query_words_dict[fruit_idx]))
                self.anomaly_flag = True

            # # todo centroid of fruit and tag_0 mismatch (if exist) ,  only one tag, center

            # patch hist
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),
                                           interpolation=cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam,
                              binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask',
                              'patch merge sam', 'binary_foreground']
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
        few_shot_samples = F.interpolate(few_shot_samples, size=(448, 448), mode=self.inter_mode,
                                         align_corners=self.align_corners, antialias=self.antialias)

        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(few_shot_samples,
                                                                                           self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0] * p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (bs, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (bs, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(
                0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size),
                                              mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1,
                                                                           self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1)  # (bsx64x64, 1024x4)

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(few_shot_samples,
                                                                     out_layer_list=self.feature_list_dinov2)  # 4 x [bs, 32x32, 1024]
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (bs, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(self.k_shot, self.ori_feat_size, self.ori_feat_size,
                                                           -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size),
                                                mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(
                self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (bsx64x64, 1024x4)

        cluster_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            cluster_features = temp_feat if cluster_features is None else torch.cat((cluster_features, temp_feat), 1)
        if self.feat_size != self.ori_feat_size:
            cluster_features = cluster_features.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2)
            cluster_features = F.interpolate(cluster_features, size=(self.feat_size, self.feat_size),
                                             mode=self.inter_mode, align_corners=self.align_corners)
            cluster_features = cluster_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(
                self.cluster_feature_id))
        cluster_features = F.normalize(cluster_features, p=2, dim=-1)

        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(
                0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size),
                                              mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        num_clusters = self.cluster_num_dict[class_name]
        _, self.cluster_centers = kmeans(X=cluster_features, num_clusters=num_clusters, device=self.device)

        self.query_obj = encode_obj_text(self.model_clip, self.query_words_dict[class_name], self.tokenizer,
                                         self.device)
        self.patch_query_obj = encode_obj_text(self.model_clip, self.patch_query_words_dict[class_name], self.tokenizer,
                                               self.device)
        self.classes = self.query_obj.shape[0]

        scores = []
        foreground_pixel_hist = []
        splicing_connectors_distance = []
        patch_token_hist = []
        mem_instance_masks = []
        grid_areas = []
        pin_areas = []
        each_grid_pins_counts = []


        for image, cluster_feature, proj_patch_token in zip(few_shot_samples.chunk(self.k_shot),
                                                                           cluster_features.chunk(self.k_shot),
                                                                           proj_patch_tokens.chunk(self.k_shot),
                                                                           ):
                                                                           #few_shot_paths):
            # path = os.path.dirname(few_shot_path).split('/')[-1] + "_" + os.path.basename(few_shot_path).split('.')[0]
            self.anomaly_flag = False
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name, None)
            if self.class_name == 'pushpins':
                patch_token_hist.append(results["clip_patch_hist"])
                grid_areas.append(results['grid_area'])
                pin_areas.append(results['pin_area'])
                each_grid_pins_counts.append(results['each_grid_pins_counts'])

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

        if len(grid_areas) != 0:
            self.grid_area = np.mean(grid_areas)
        if len(pin_areas) != 0:
            self.pin_area = np.mean(pin_areas)
        if len(each_grid_pins_counts) != 0:
            flattened_grid_pixel_hist = np.concatenate(each_grid_pins_counts)
            self.each_grid_pins_counts = np.mean(flattened_grid_pixel_hist)

        if len(foreground_pixel_hist) != 0:
            self.foreground_pixel_hist = np.mean(foreground_pixel_hist)
        if len(splicing_connectors_distance) != 0:
            self.splicing_connectors_distance = np.mean(splicing_connectors_distance)
        if len(patch_token_hist) != 0:  # patch hist
            self.patch_token_hist = np.stack(patch_token_hist)
        if len(mem_instance_masks) != 0:
            self.mem_instance_masks = mem_instance_masks

        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset


    def process(self, class_name: str, few_shot_samples: list[torch.Tensor]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device)
        scores, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(
            class_name, few_shot_samples)

    def process_subcategory(self, cat_id, normal_dino_patches, heatmap_cache, stage="test", k=1):
        """处理单个子类别的优化版本"""

        # 使用生成器避免中间变量累积
        def get_features():
            for j in range(k):
                path = f"./outputs/{self.class_name}/{stage}/{j:03d}/mask_{cat_id}.png"
                mask = heatmap_cache[path]

                _, thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = torch.from_numpy(thresh).view(1, 1, *thresh.shape)
                mask_stack = F.interpolate(thresh, size=self.perform_grid_size, mode="bilinear",
                                           align_corners=True).reshape(self.perform_grid_size ** 2)
                if mask_stack.sum() < 1:
                    selected = None
                else:
                    selected = normal_dino_patches[j][mask_stack]
                yield selected

        # 流式处理特征
        sub_features, sub_patch_features = [], []
        for feat in get_features():
            if feat is None: continue
            sub_patch_features.append(feat)
            sub_features.append(feat.mean(dim=0))
        if len(sub_features) > 0:
            # 延迟合并
            sub_patch_features = torch.cat(sub_patch_features, dim=0)
            sub_features = torch.stack(sub_features, dim=0)

            # 原地归一化
            F.normalize(sub_patch_features, p=2, dim=-1, out=sub_patch_features)
            F.normalize(sub_features, p=2, dim=1, out=sub_features)

        return sub_features, sub_patch_features


    def generate_mask(self, sample):
        masks, boxes, obj_names, pil_img = self.mask_generator.forward(sample, self.class_name)     # SAM mask
        clusted_masks, label_ids = self.mask_generator.cluster_mask(masks, boxes, obj_names, self.class_name)
        if self.class_name == "screw_bag":
            clusted_masks[-1] = clusted_masks[-1] & (~clusted_masks[-2].astype(bool)).astype(int) * 255
            binary = clusted_masks[-1]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 300: # 448x448 300
                    binary[temp_mask] = 0 # set to background
            clusted_masks[-1] = binary
        return clusted_masks, boxes, obj_names, label_ids, pil_img



    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        few_shot_samples = setup_data.get("few_shot_samples")
        class_name = setup_data.get("dataset_category")
        #few_shot_paths = setup_data.get("few_shot_samples_path")
        self.class_name = class_name

        if self.class_name == "screw_bag" or self.class_name == "breakfast_box" or self.class_name == "juice_bottle":
            if not self.gdino_init:
                self.mask_generator = GSAM2Predictor()
                self.gdino_init = True
            self.shot = len(few_shot_samples)  # k shots
            transformed_normal_image = self.transform_dino(few_shot_samples).cuda()
            heatmap_cache = dict()
            if self.gdino_init:
                self.mask_generator.objs_dict = self.objs_dict
                self.mask_generator.foreground_num = self.foreground_num
                all_area_hists = list()
                all_color_hists = list()
                for idx, sample in enumerate(few_shot_samples):
                    clusted_masks, boxes, obj_names, label_ids, pil_img = self.generate_mask(sample)
                    area_hist = list()
                    color_hist = list()
                    # 转换为LAB色彩空间
                    imglabo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2LAB)
                    color_a = imglabo[:, :, 1].astype(np.float32)  # 使用float32提高精度
                    color_b = imglabo[:, :, 2].astype(np.float32)
                    epsilon = 1e-7  # 更小的除数保护值
                    color_ratio_sq = (color_b / (color_a + epsilon)) ** 2
                    for cat_id, mask in enumerate(clusted_masks):
                        heatmap_cache[f"./outputs/{self.class_name}/train/{idx:03d}/mask_{cat_id}.png"] = mask
                        area = mask.sum()
                        area_hist.append(area)
                        masked_color = color_ratio_sq * mask
                        color_value = np.sum(masked_color) / (area + epsilon)
                        color_hist.append(color_value)
                    all_area_hists.append(area_hist)
                    all_color_hists.append(color_hist)

                    if self.vis:
                        path = f"./outputs/{self.class_name}/train/{idx:03d}/img.png"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        pil_img.save(path)
                        for cat_id, mask in enumerate(clusted_masks):
                            path = f"./outputs/{self.class_name}/train/{idx:03d}/mask_{cat_id}.png"

                            cv2.imwrite(path, mask * 255)
                # 归一化结果
                self.class_hists_train = np.mean(np.vstack(all_area_hists), axis=0)
                self.class_color_train = np.mean(np.vstack(all_color_hists), axis=0)

            with torch.no_grad():
                self.dinov2_normal_patch_tokens = self.model_dinov2_yy.forward_intermediates(transformed_normal_image, indices=self.out_layers, intermediates_only=True)
                self.dinov2_normal_patch_tokens = [ F.interpolate(p, size=self.perform_grid_size, mode="bilinear", align_corners=True) for p in self.dinov2_normal_patch_tokens]
                self.dinov2_normal_patch_tokens = [p.permute(0, 2, 3, 1).view(self.shot, self.perform_grid_size ** 2, self.embedding_dim) for p in self.dinov2_normal_patch_tokens]

                self.eva02_normal_patch_tokens = self.model_eva02.forward_intermediates(transformed_normal_image, indices=self.out_layers, intermediates_only=True)
                self.eva02_normal_patch_tokens = [F.interpolate(p, size=self.perform_grid_size, mode="bilinear", align_corners=True) for p in self.eva02_normal_patch_tokens]
                self.eva02_normal_patch_tokens = [p.permute(0, 2, 3, 1).view(self.shot, self.perform_grid_size ** 2, self.embedding_dim) for p in self.eva02_normal_patch_tokens]

            self.selected_features_eva02 = []
            self.part_normal_patch_tokens_eva02 = []
            for layer in range(4):
                selected_features_eva02_layer = list()
                part_normal_patch_tokens_eva02_layer = list()
                for cat_id in range(self.foreground_num[self.class_name]):
                    normal_patch_tokens_eva02 = self.eva02_normal_patch_tokens[layer]
                    sub_feat, sub_patch = self.process_subcategory(
                        cat_id, normal_patch_tokens_eva02, heatmap_cache, "train", self.shot
                    )
                    selected_features_eva02_layer.append(sub_feat)
                    part_normal_patch_tokens_eva02_layer.append(sub_patch)
                self.part_normal_patch_tokens_eva02.append(part_normal_patch_tokens_eva02_layer)  # [4, cats, ]
                self.selected_features_eva02.append(selected_features_eva02_layer)

            self.selected_features_dinov2 = []
            self.part_normal_patch_tokens_dinov2 = []
            for layer in range(4):
                selected_features_dinov2_layer = list()
                part_normal_patch_tokens_dinov2_layer = list()
                for cat_id in range(self.foreground_num[self.class_name]):
                    normal_patch_tokens_dinov2 = self.dinov2_normal_patch_tokens[layer]
                    sub_feat, sub_patch = self.process_subcategory(
                        cat_id, normal_patch_tokens_dinov2, heatmap_cache, "train", self.shot
                    )
                    selected_features_dinov2_layer.append(sub_feat)
                    part_normal_patch_tokens_dinov2_layer.append(sub_patch)
                self.part_normal_patch_tokens_dinov2.append(part_normal_patch_tokens_dinov2_layer)  # [4, cats, ]
                self.selected_features_dinov2.append(selected_features_dinov2_layer)

            del heatmap_cache

        else:
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

    @property
    def batch_size(self) -> int:
        """Batch size of the model."""
        # TODO: Reduce the batch size in case your model is too large to fit in memory.
        return 1

    def forward_one_sample(self, batch: torch.Tensor, mem_patch_feature_clip_coreset: torch.Tensor,
                           mem_patch_feature_dinov2_coreset: torch.Tensor):

        bs = batch.shape[0]
        with torch.no_grad():
            time1 = time.time()
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(batch, self.feature_list)
            time2 = time.time()
            print("clip time:", time2 - time1)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0] * p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (1, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (1, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1,
                                                                                                              2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size),
                                              mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1,
                                                                           self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1)  # (1x64x64, 1024x4)

        with torch.no_grad():
            time1 = time.time()
            patch_tokens_dinov2 = self.model_dinov2.forward_features(batch, out_layer_list=self.feature_list)
            time2 = time.time()
            print("dinov2:", time2 - time1)
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3,
                                                                                                                  1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size),
                                                mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(
                self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (1x64x64, 1024x4)

        '''adding for kmeans seg '''
        if self.feat_size != self.ori_feat_size:
            #proj_patch_tokens = proj_patch_tokens.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1,2)
            proj_patch_tokens = proj_patch_tokens.view(bs, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size),
                                              mode=self.inter_mode, align_corners=self.align_corners)
        
            #proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(self.feat_size * self.feat_size, self.embed_dim)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        mid_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            mid_features = temp_feat if mid_features is None else torch.cat((mid_features, temp_feat), -1)

        if self.feat_size != self.ori_feat_size:
            mid_features = mid_features.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            mid_features = F.interpolate(mid_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode,
                                         align_corners=self.align_corners)
            mid_features = mid_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        mid_features = F.normalize(mid_features, p=2, dim=-1)

        time1 = time.time()
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name, None)
        time2 = time.time()
        print("histogram time:", time2 - time1)

        hist_score = results['score']

        '''calculate patchcore'''
        anomaly_maps_patchcore = []

        if self.class_name in ['pushpins', 'screw_bag']:  # clip feature for patchcore
            len_feature_list = len(self.feature_list)
            for patch_feature, mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1),
                                                        mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy()  # 1: normal 0: abnormal
                anomaly_map_patchcore = 1 - normal_map_patchcore

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        if self.class_name in ['splicing_connectors', 'breakfast_box', 'juice_bottle']:  # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for patch_feature, mem_patch_feature in zip(patch_tokens_dinov2.chunk(len_feature_list, dim=-1),
                                                        mem_patch_feature_dinov2_coreset.chunk(len_feature_list,
                                                                                               dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy()  # 1: normal 0: abnormal
                anomaly_map_patchcore = 1 - normal_map_patchcore

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        structural_score = np.stack(anomaly_maps_patchcore).mean(0).max()
        anomaly_map_structural = np.stack(anomaly_maps_patchcore).mean(0).reshape(self.feat_size, self.feat_size)

        if self.class_name == 'pushpins':
            results = {'hist_score': hist_score, 'structural_score': structural_score, "anomaly_map_structural": anomaly_map_structural}

        else:
            instance_masks = results["instance_masks"]
            anomaly_instances_hungarian = []
            instance_hungarian_match_score = 1.
            if self.mem_instance_masks is not None and len(instance_masks) != 0:
                for patch_feature, batch_mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1),
                                                                mem_patch_feature_clip_coreset.chunk(len_feature_list,
                                                                                                    dim=-1)):
                    instance_features = [patch_feature[mask, :].mean(0, keepdim=True) for mask in instance_masks]
                    instance_features = torch.cat(instance_features, dim=0)
                    instance_features = F.normalize(instance_features, dim=-1)
                    mem_instance_features = []
                    for mem_patch_feature, mem_instance_masks in zip(batch_mem_patch_feature.chunk(self.k_shot),
                                                                    self.mem_instance_masks):
                        mem_instance_features.extend(
                            [mem_patch_feature[mask, :].mean(0, keepdim=True) for mask in mem_instance_masks])
                    mem_instance_features = torch.cat(mem_instance_features, dim=0)
                    mem_instance_features = F.normalize(mem_instance_features, dim=-1)

                    normal_instance_hungarian = (instance_features @ mem_instance_features.T)
                    cost_matrix = (1 - normal_instance_hungarian).cpu().numpy()

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    cost = cost_matrix[row_ind, col_ind].sum()
                    cost = cost / min(cost_matrix.shape)
                    anomaly_instances_hungarian.append(cost)

                instance_hungarian_match_score = np.mean(anomaly_instances_hungarian)

            results = {'hist_score': hist_score, 'structural_score': structural_score,
                    'instance_hungarian_match_score': instance_hungarian_match_score,
                    "anomaly_map_structural": anomaly_map_structural}

        return results

    def compute_patch_similarity(self, query_feats, gallery_feats, feat_dim):
        assert query_feats.shape[-1] == feat_dim, f"特征维度应为{feat_dim}"
        assert gallery_feats.shape[-1] == feat_dim, f"特征维度应为{feat_dim}"

        query = F.normalize(query_feats, dim=-1)
        gallery = F.normalize(gallery_feats, dim=-1)
        sim_matrix = (query @ gallery.T)

        return sim_matrix.max(dim=1)[0]

    def generate_heatmap(self, scores):
        """生成异常热图
        Args:
            scores: 原始得分 [H,W]
            original_size: 目标输出尺寸
        Returns:
            热图张量 [1,1,H,W]
        """
        # 调整到16x16基础尺寸
        base_size = int(scores.numel() ** 0.5)
        scores = scores.view(1, 1, base_size, base_size)

        return scores

    @torch.no_grad()
    def compute_anomaly_maps(self, test_patch_tokens, normal_patch_tokens):
        test_sim_all = []
        for layer_idx, (query_layer, normal_layer) in enumerate(zip(
                test_patch_tokens,
                normal_patch_tokens
        )):
            # 当前层特征处理
            query = query_layer.reshape(-1, self.embedding_dim)
            gallery = normal_layer.reshape(-1, self.embedding_dim)

            # 单层相似度计算
            layer_sim = self.compute_patch_similarity(query, gallery, self.embedding_dim)
            test_sim_all.append(layer_sim)

            # 及时释放中间变量
            del query, gallery, layer_sim

        test_sim = torch.mean(torch.stack(test_sim_all), dim=0)
        anomaly_map_test = 1 - self.generate_heatmap(test_sim)

        return anomaly_map_test

    def calculate_subcategory_metrics(
            self,
            test_patch_tokens,
            heatmap_cache,
            normal_feats,
            normal_patch_tokens,
            num_layers=4
    ):
        subcategory_mean_dists = []
        subcategory_patch_sims = []
        start_index = 0
        if self.class_name == "screw_bag":
            start_index = 1

        for layer in range(num_layers):
            subcategory_mean_dists_layer = []
            subcategory_patch_sims_layer = []

            for cat_id in range(start_index, self.foreground_num[self.class_name]):
                subcategory_test_feat, subcategory_test_patch = self.process_subcategory(
                    cat_id, test_patch_tokens[layer], heatmap_cache
                )

                if len(subcategory_test_feat) == 0:
                    continue
                if len(normal_feats[layer][cat_id]) == 0:
                    continue
                dist = 1 - (subcategory_test_feat @ normal_feats[layer][cat_id].T).max().item()
                subcategory_mean_dists_layer.append(dist)

                subcategory_test_patch_reshaped = subcategory_test_patch.reshape(-1, self.embedding_dim)
                subcategory_normal_patch_reshaped = normal_patch_tokens[layer][cat_id].reshape(-1, self.embedding_dim)
                cosine_similarity_matrix = (subcategory_test_patch_reshaped @ subcategory_normal_patch_reshaped.T)
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                subcategory_patch_sims_layer.append(sim_max)

            # 合并当前层的补丁相似度
            if subcategory_patch_sims_layer:
                subcategory_patch_sims_layer = torch.cat(subcategory_patch_sims_layer, dim=0)
                subcategory_patch_sims.append(subcategory_patch_sims_layer)
                subcategory_mean_dists.append(subcategory_mean_dists_layer)

        # 计算跨层的平均补丁相似度
        if subcategory_patch_sims:
            subcategory_patch_sims = torch.mean(torch.stack(subcategory_patch_sims, dim=0), dim=0)
            anomaly_map_ret_part = 1 - subcategory_patch_sims
        else:
            anomaly_map_ret_part = None

        # 计算子类别平均距离
        if subcategory_mean_dists:
            subcategory_mean_dists = torch.tensor(subcategory_mean_dists).reshape(-1, len(subcategory_mean_dists[0]))
            subcategory_mean_dists = torch.mean(subcategory_mean_dists, dim=0)
        else:
            subcategory_mean_dists = None

        return anomaly_map_ret_part, subcategory_mean_dists

    def save_results_to_csv(image_path,
                            anomaly_map_ret_dinov2,
                            anomaly_map_ret_eva02,
                            dinov2_anomaly_map_ret_part,
                            dinov2_subcategory_mean_dists,
                            eva02_anomaly_map_ret_part,
                            eva02_subcategory_mean_dists,
                            area_hist_avg, color_hist_avg,
                            flag,
                            output_path="results.csv"):

        # 准备CSV文件名(带时间戳)
        csv_filename = output_path

        # 检查文件是否存在，决定是否写入表头
        write_header = not os.path.exists(csv_filename)

        # 准备数据行
        row_data = {
            "image_path": image_path,
            "anomaly_map_ret_dinov2": anomaly_map_ret_dinov2,
            "anomaly_map_ret_eva02": anomaly_map_ret_eva02,
            "dinov2_anomaly_map_ret_part": dinov2_anomaly_map_ret_part,
            "dinov2_subcategory_mean_dists": dinov2_subcategory_mean_dists,
            "eva02_anomaly_map_ret_part": eva02_anomaly_map_ret_part,
            "eva02_subcategory_mean_dists": eva02_subcategory_mean_dists,
            "flag": flag,
        }

        # 添加area_hist_avg的每个元素作为单独列
        for i, val in enumerate(area_hist_avg):
            row_data[f"area_hist_avg_{i}"] = val

        # 添加color_hist_avg的每个元素作为单独列
        for i, val in enumerate(color_hist_avg):
            row_data[f"color_hist_avg_{i}"] = val

        # 写入CSV文件
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = row_data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow(row_data)

    def calculate_hole_radii_connected_components(self, mask_image):
        mask_image = cv2.resize(mask_image, (1600, 1100), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((5, 5), np.uint8)
        mask_image = cv2.erode(mask_image, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
        min_size = 300
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                labels[labels == i] = 0
        results = []
        if num_labels < 5:
            return results

        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            outer_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
            if not outer_contours:
                continue

            # 孔洞轮廓（hierarchy[i][3]!=-1）
            hole_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]
            hole_radius = 0.0

            if hole_contours:
                # 取面积最大的孔洞（假设主空心区域）
                largest_hole = max(hole_contours, key=cv2.contourArea)
                hole_mask = np.zeros_like(component_mask)
                cv2.drawContours(hole_mask, [largest_hole], -1, 255, -1)

                # 距离变换法求最大内接圆半径
                dist_transform = cv2.distanceTransform(hole_mask, cv2.DIST_L2, 3)
                _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
                hole_radius = max_val

            results.append(float(hole_radius))
        return results
    def compute_juice_color_area(self, hsv,):
        COLOR_HSV_RANGES = {
            'red': [([0, 100, 50], [10, 255, 200]),],  # 红色有两个H区段
            'yellow': [([20, 80, 80], [45, 255, 255])],
            'milky_white': [([15, 5, 80], [80, 100, 255])]  # 乳白色：低饱和度，高亮度
        }
        max_height = 0
        detected_color = 'unknown'
        liquid_top = 1600
        for color_name, hsv_ranges in COLOR_HSV_RANGES.items():
            for lower, upper in hsv_ranges:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask = cv2.inRange(hsv, lower_np, upper_np)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    if h > max_height:
                        detected_color = color_name
                        liquid_top = y
                        max_height = h
        return liquid_top, detected_color
    
    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # TODO: Implement the forward pass of the model.
        '''
        batch_size = image.shape[0]
        return ImageBatch(
            image=image,
            pred_score=torch.zeros(batch_size, device=image.device),
        )
        '''
        if self.class_name == "screw_bag" or self.class_name == "breakfast_box" or self.class_name == "juice_bottle":
            self.scew_bag_abnormal_flag = 0
            self.breakfast_box_abnormal_flag = 0
            self.bottle_abnormal_flag = 0

            heatmap_cache = dict()
            area_hist = list()
            color_hist = list()
            if self.gdino_init:
                clusted_masks, boxes, obj_names, label_ids, pil_img = self.generate_mask(batch[0])
                if self.class_name == "breakfast_box":
                    rgb_img = np.array(pil_img)
                    binary = clusted_masks[4]
                    avg_color = []
                    if np.sum(binary > 0) > 100:
                        for channel in range(rgb_img.shape[2]):
                            channel_avg = np.mean(rgb_img[:, :, channel][binary > 0])
                            avg_color.append(float(channel_avg))
                        if avg_color[0] > 160 or avg_color[0] < 90 or avg_color[2] > 65 or avg_color[2] < 40:

                            self.breakfast_box_abnormal_flag = 1

                if self.class_name == "screw_bag":
                    binary = clusted_masks[-1]
                    if np.sum(binary > 0) > 100:
                        mask_image = cv2.resize(binary, (1600, 1100), interpolation=cv2.INTER_NEAREST)
                        kernel = np.ones((3, 3), np.uint8)
                        mask_image = cv2.erode(mask_image, kernel, iterations=1)
                        contours, _ = cv2.findContours(
                            mask_image.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        if len(contours) >= 2:
                            lengths = []
                            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                            for contour in sorted_contours:
                                rotated_rect = cv2.minAreaRect(contour)
                                width, height = rotated_rect[1]
                                lengths.append(max(width, height))
                            longer = max(lengths)
                            shorter = min(lengths)
                            ratio = (longer - shorter) / (longer + 1e-7)
                            if ratio < 0.25:
                                self.scew_bag_abnormal_flag = 1
                    # 螺母和垫片
                    binary = clusted_masks[-2]
                    if np.sum(binary > 0) > 100:
                        results = self.calculate_hole_radii_connected_components(binary)
                        if len(results) > 0:
                            results.sort(reverse=True)
                            circles = list()
                            nuts = list()
                            for i, res in enumerate(results):
                                if res < 10:
                                    continue
                                if res > 30:
                                    circles.append(i)
                                else:
                                    nuts.append(i)
                            if (len(circles) + len(nuts) >= 4) and len(circles) != 2 or len(nuts) != 2:
                                self.scew_bag_abnormal_flag = 1
                if self.class_name == "juice_bottle":
                    # 标签个数检查
                    binary = clusted_masks[-1]
                    binary = cv2.resize(binary, (800, 1600), interpolation=cv2.INTER_NEAREST)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

                    for i in range(1, num_labels):
                        temp_mask = labels == i
                        if np.sum(temp_mask) <= 300: # 448x448 64
                            binary[temp_mask] = 0 # set to background
                    tag_num_labels, labels_tag, stats_tag, centroids_tag = cv2.connectedComponentsWithStats(binary)
                    if tag_num_labels >= 3:
                        sorted_indices = np.argsort(stats_tag[1:, cv2.CC_STAT_AREA])[::-1]
                        largest_labels = sorted_indices[:2] + 1
                        # 标签距离有问题， 打印质心的距离
                        dist_x = list()
                        dist_y = list()
                        for i in largest_labels:
                            # 计算两个标签的质心距离
                            if i < 1 or i >= num_labels:
                                continue
                            centroid = centroids_tag[i]
                            dist_x.append(centroid[0])
                            dist_y.append(centroid[1])
                        x_dist = abs(dist_x[0] - dist_x[1])
                        y_dist = abs(dist_y[0] - dist_y[1])
                        if x_dist > 50 or y_dist > 520 or y_dist < 350:
                            # print(f"{abs(dist_x[0] - dist_x[1])} {abs(dist_y[0] - dist_y[1])} 标签距离异常")
                            self.bottle_abnormal_flag = 1
                        # 标签互换
                        if dist_y[0] > dist_y[1]:
                            # print(f" {dist_y[0]} {dist_y[1]} 标签互换")
                            self.bottle_abnormal_flag = 1
                    else:
                        # print(f"标签个数异常: {tag_num_labels}")
                        self.bottle_abnormal_flag = 1
                    binary = clusted_masks[-2]
                    binary = cv2.resize(binary, (800, 1600), interpolation=cv2.INTER_NEAREST)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

                    for i in range(1, num_labels):
                        temp_mask = labels == i
                        if np.sum(temp_mask) <= 300: # 448x448 64
                            binary[temp_mask] = 0 # set to background
                    num_labels_fruit, labels_fruit, stats_fruit, centroids_fruit = cv2.connectedComponentsWithStats(binary)

                    if num_labels_fruit < 2:
                        # print(f"with {num_labels - 1} 水果没有异常.")
                        self.bottle_abnormal_flag = 1
                    elif num_labels_fruit >= 2 and tag_num_labels >= 3:
                        # 水果位置异常
                        sorted_indices = np.argsort(stats_tag[1:, cv2.CC_STAT_AREA])[::-1]
                        largest_idx = sorted_indices[:1] + 1
                        # 标签质心
                        c1 = centroids_tag[largest_idx[0]]
                        # 水果质心
                        c2 = centroids_fruit[1]
                        # num_labels_fruit >2 求平均质心
                        if num_labels_fruit > 2:
                            c2 = np.mean(centroids_fruit[1:], axis=0)
                        # 计算距离
                        dist_x = abs(c1[0] - c2[0])
                        dist_y = abs(c1[1] - c2[1])
                        if dist_x > 30 or dist_y > 36:
                            # print(f"{dist_x} {dist_y} 水果位置异常")
                            self.bottle_abnormal_flag = 1
                    binary = clusted_masks[0]

                    binary = cv2.resize(binary, (800, 1600), interpolation=cv2.INTER_NEAREST)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

                    for i in range(1, num_labels):
                        temp_mask = labels == i
                        if np.sum(temp_mask) <= 300: # 448x448 64
                            binary[temp_mask] = 0 # set to background
                    rgb_img = np.array(pil_img)
                    rgb_img = cv2.resize(rgb_img, (800, 1600), interpolation=cv2.INTER_LINEAR)
                    avg_color = []
                    if np.sum(binary > 0) > 100:
                        for channel in range(rgb_img.shape[2]):
                            channel_avg = np.mean(rgb_img[:,:,channel][binary > 0])
                            avg_color.append(float(channel_avg))
                        if avg_color[0] < 95 :
                            # print(f"average color: {avg_color}")
                            self.bottle_abnormal_flag = 1
                        if not self.bottle_abnormal_flag:
                            # 液体面积和颜色检查
                            hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
                            hsv = cv2.bitwise_and(hsv, hsv, mask=binary)
                            liquid_top, detected_color = self.compute_juice_color_area(hsv)
                            if liquid_top < 1600 and detected_color in ["red", "yellow", "milky_white"]:
                                if liquid_top > 440 or liquid_top < 370:
                                    # print(f"img_path {image_path} 液体高度异常: {liquid_top}")
                                    self.bottle_abnormal_flag = 1
                                # 检查颜色和水果是否匹配
                                if len(label_ids[1]) > 0:
                                    pred_classes = [ obj_names[i] for i in label_ids[1]]
                                    pred_classes = set(pred_classes)
                                    # pred_classes = [ tag.split(" ")[0] for tag in pred_classes if " " in tag]
                                    # print(pred_classes)
                                    if detected_color == "red" and "cherry" not in pred_classes:
                                        # print(f"img_path {image_path} 红色液体异常: {detected_color} {pred_classes}")
                                        self.bottle_abnormal_flag = 1
                                    elif detected_color == "yellow" and "orange" not in pred_classes:
                                        # print(f"img_path {image_path} 黄色液体异常: {detected_color} {pred_classes}")
                                        self.bottle_abnormal_flag = 1
                                    elif detected_color == "milky_white" and "banana" not in pred_classes:
                                        # print(f"img_path {image_path} 乳白色液体异常: {detected_color} {pred_classes}")
                                        self.bottle_abnormal_flag = 1

                imglabo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2LAB)
                color_a = imglabo[:, :, 1].astype(np.float32)
                color_b = imglabo[:, :, 2].astype(np.float32)
                epsilon = 1e-7
                color_ratio_sq = (color_b / (color_a + epsilon)) ** 2
                for cat_id, mask in enumerate(clusted_masks):
                    heatmap_cache[f"./outputs/{self.class_name}/test/000/mask_{cat_id}.png"] = mask
                    area = mask.sum()
                    area_hist.append(area)
                    masked_color = color_ratio_sq * mask
                    color_value = np.sum(masked_color) / (area + epsilon)
                    color_hist.append(color_value)


            area_hist_avg = np.array(area_hist) / self.class_hists_train
            color_hist_avg = np.array(color_hist) / self.class_color_train

            transformed_image = self.transform_dino(batch)
            test_patch_tokens_dinov2 = self.model_dinov2_yy.forward_intermediates(transformed_image,
                                                                               indices=self.out_layers,
                                                                               intermediates_only=True)
            test_patch_tokens_dinov2 = [
                F.interpolate(p, size=self.perform_grid_size, mode="bilinear", align_corners=True) for p in
                test_patch_tokens_dinov2]
            test_patch_tokens_dinov2 = [p.permute(0, 2, 3, 1).view(-1, self.perform_grid_size ** 2, self.embedding_dim)
                                        for p in test_patch_tokens_dinov2]
            anomaly_map_ret_dinov2 = self.compute_anomaly_maps(test_patch_tokens_dinov2,
                                                               self.dinov2_normal_patch_tokens)

            test_patch_tokens_eva02 = self.model_eva02.forward_intermediates(transformed_image, indices=self.out_layers,
                                                                             intermediates_only=True)
            test_patch_tokens_eva02 = [
                F.interpolate(p, size=self.perform_grid_size, mode="bilinear", align_corners=True) for p in
                test_patch_tokens_eva02]
            test_patch_tokens_eva02 = [p.permute(0, 2, 3, 1).view(-1, self.perform_grid_size ** 2, self.embedding_dim)
                                       for p in test_patch_tokens_eva02]
            anomaly_map_ret_eva02 = self.compute_anomaly_maps(test_patch_tokens_eva02, self.eva02_normal_patch_tokens)

            dinov2_anomaly_map_ret_part, dinov2_subcategory_mean_dists = self.calculate_subcategory_metrics(
                test_patch_tokens_dinov2, heatmap_cache, self.selected_features_dinov2,
                self.part_normal_patch_tokens_dinov2
            )
            eva02_anomaly_map_ret_part, eva02_subcategory_mean_dists = self.calculate_subcategory_metrics(
                test_patch_tokens_eva02, heatmap_cache, self.selected_features_eva02,
                self.part_normal_patch_tokens_eva02
            )

            if self.class_name == "screw_bag":
                return ImageBatch(
                    image = batch,
                    pred_score =  torch.tensor(
                    20 * anomaly_map_ret_dinov2.max().item() +
                    10 * anomaly_map_ret_eva02.max().item() +
                    10 * dinov2_anomaly_map_ret_part.max().item() +
                    10 * dinov2_subcategory_mean_dists.max().item() +
                    10 * eva02_anomaly_map_ret_part.max().item() +
                    10 * eva02_subcategory_mean_dists.max().item() +
                    25 * abs(area_hist_avg[1] - 1) +
                    25 * abs(area_hist_avg[2] - 1) +
                    15 * self.scew_bag_abnormal_flag
                    ).to(self.device)
                )
            elif self.class_name == "juice_bottle":            
                return  ImageBatch(
                    image = batch,
                    pred_score = torch.tensor( 
                        10 * anomaly_map_ret_eva02.max().item() +  
                        5 * dinov2_subcategory_mean_dists.max().item() +
                        10 * eva02_anomaly_map_ret_part.max().item() +
                        10 * eva02_subcategory_mean_dists.max().item() +
                        5 * self.bottle_abnormal_flag
                        ).to(self.device)
                    )
            else:
                return ImageBatch(
                    image = batch,
                    pred_score = torch.tensor(
                        30 * anomaly_map_ret_dinov2.max().item() +
                        30 * anomaly_map_ret_eva02.max().item() +
                        10 * abs(area_hist_avg.min() - 1) +
                        5 * abs(area_hist_avg.max() - 1) +
                        5 * abs(color_hist_avg.min() - 1) +
                        5 * abs(color_hist_avg[-1] - 1) +
                        5 * self.breakfast_box_abnormal_flag
                    ).to(self.device)
                )


        else:
            self.anomaly_flag = False
            batch = self.transform(batch).to(self.device)
            results = self.forward_one_sample(batch, self.mem_patch_feature_clip_coreset,
                                              self.mem_patch_feature_dinov2_coreset)

            hist_score = results['hist_score']
            structural_score = results['structural_score']
            # standardization
            standard_structural_score = (structural_score - self.stats[self.class_name]["structural_scores"]["mean"]) / \
                                        self.stats[self.class_name]["structural_scores"]["unbiased_std"]
            if self.class_name == 'pushpins':
                pred_score = standard_structural_score

            else:
                instance_hungarian_match_score = results['instance_hungarian_match_score']
                standard_instance_hungarian_match_score = (instance_hungarian_match_score -
                                            self.stats[self.class_name]["instance_hungarian_match_scores"][
                                            "mean"]) / \
                                            self.stats[self.class_name]["instance_hungarian_match_scores"][
                                            "unbiased_std"]

                standard_hist_score = (hist_score - self.stats[self.class_name]["hist_scores"]["mean"]) / self.stats[self.class_name]["hist_scores"]["unbiased_std"]
                if self.class_name == 'breakfast_box':
                    a1 = 16/(16+4+19)
                    a2 = 4/(16+4+19)
                    a3 = 19/(16+4+19)
                    pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score

                elif self.class_name == 'juice_bottle':
                    # a1 = 1/(1+7+0)
                    # a2 = 7/(1+7+0)
                    # a3 = 0
                    a1 = 1/(1+5+0)
                    a2 = 5/(1+5+0)
                    a3 = 0
                    pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
                elif self.class_name == 'screw_bag':
                    a1 = 1/(1+2+16)
                    a2 = 2/(1+2+16)
                    a3 = 16/(1+2+16)
                    pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
                else:
                    a1 = 1
                    a2 = 0
                    a3 = 0
                    pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
            
                    # pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)

            def sigmoid(z):
                return 1 / (1 + np.exp(-z))
            pred_score = sigmoid(pred_score)
            if self.anomaly_flag:
                pred_score = 1.
                self.anomaly_flag = False

            batch_size = batch.shape[0]
            pred_score = torch.tensor(pred_score).to(self.device)
            return ImageBatch(image=batch, pred_score=pred_score,)



class TensorDictDB:
    def __init__(self, class_name: str):
        filename = class_name + '.h5'
        self.filename = filename
        if os.path.exists(filename):
            self._validate_existing_db()
        else:
            self._initialize_db()

    def _initialize_db(self):
        """初始化空数据库结构"""
        with h5py.File(self.filename, 'w') as f:
            f.create_group('hash_index')
            f.create_group('tensors')
            f.create_group('metadata')

    def _validate_existing_db(self):
        """验证现有数据库结构完整性"""
        with h5py.File(self.filename, 'r') as f:
            required_groups = ['hash_index', 'tensors', 'metadata']
            for group in required_groups:
                if group not in f:
                    raise ValueError(f"损坏的数据库文件：缺少 {group} 分组")

    def _tensor_hash(self, tensor: np.ndarray) -> str:
        """生成张量的哈希（支持NumPy和PyTorch CPU/CUDA张量）"""
        if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            tensor = tensor.cpu().numpy()
        return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()

    def _save_metadata_list(self, group: h5py.Group, data_list: List[Dict[str, Any]]):
        """保存字典列表到HDF5 Group"""
        for idx, data in enumerate(data_list):
            element_group = group.create_group(f"element_{idx}")
            self._save_metadata(element_group, data)

    def _save_metadata(self, group: h5py.Group, data: Dict[str, Any]):
        """递归保存单个字典到HDF5 Group"""
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            elif isinstance(value, (np.ndarray, list)):
                arr = np.array(value) if isinstance(value, list) else value
                group.create_dataset(key, data=arr, compression='gzip')
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_metadata(subgroup, value)
            else:
                raise TypeError(f"不支持的类型: {type(value)}")

    def _load_metadata_list(self, group: h5py.Group) -> List[Dict[str, Any]]:
        """从HDF5 Group加载字典列表"""
        return [self._load_metadata(group[name]) for name in sorted(group.keys())]

    def _load_metadata(self, group: h5py.Group) -> Dict[str, Any]:
        """递归加载单个字典"""
        metadata = {}
        # 加载属性
        for key, value in group.attrs.items():
            metadata[key] = value
        # 加载数据集和子组
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                metadata[key] = item[()]
            elif isinstance(item, h5py.Group):
                metadata[key] = self._load_metadata(item)
        return metadata

    def get(self, tensor_key: np.ndarray) -> List[Dict[str, Any]]:
        """查询Key，返回对应的字典列表（不存在则返回None）"""
        with h5py.File(self.filename, 'r') as f:
            hash_key = self._tensor_hash(tensor_key)
            if hash_key in f['hash_index']:
                meta_ref = f['hash_index'][hash_key][()]
                return self._load_metadata_list(f[meta_ref])
        return None

    def add(self, tensor_key: np.ndarray, meta_list: List[Dict[str, Any]]) -> bool:
        """添加Key-Value列表（Key不存在时返回True）"""
        hash_key = self._tensor_hash(tensor_key)
        with h5py.File(self.filename, 'a') as f:
            if hash_key in f['hash_index']:
                return False  # Key已存在

            # 存储张量
            tensor_ref = f'tensors/{hash_key}'
            f.create_dataset(tensor_ref, data=tensor_key, compression='gzip')

            # 存储元数据列表
            meta_ref = f'metadata/{hash_key}'
            meta_group = f.create_group(meta_ref)
            self._save_metadata_list(meta_group, meta_list)

            # 更新索引
            f['hash_index'].create_dataset(hash_key, data=meta_ref)
            return True

    def get_or_add(self, tensor_key: np.ndarray, meta_list: List[Dict[str, Any]] = None):
        """查询Key，不存在时添加并返回None，存在时返回字典列表"""
        existing = self.get(tensor_key)
        if existing is not None:
            return existing
        elif meta_list is not None:
            self.add(tensor_key, meta_list)
            return None
        raise ValueError("Key不存在且未提供meta_list")
