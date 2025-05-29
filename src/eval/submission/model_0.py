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

# Grounding DINO
from eval.submission.grounded_sam import (
    load_model,
    get_grounding_output
)
from groundingdino.util import get_tokenlizer
from groundingdino.models import build_model


def to_np_img(m):
    m = m.permute(1, 2, 0).cpu().numpy()
    mean = np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    std = np.array([[[0.26862954, 0.26130258, 0.27577711]]])
    m  = m * std + mean
    return np.clip((m * 255.), 0, 255).astype(np.uint8)


class Model(nn.Module):
    """TODO: Implement your model here"""

    def __init__(self) -> None:
        super().__init__()

        # setup_seed(42) # 提交版本注释
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

        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        
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
        self.model_sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.model_sam)
        
        GROUNDING_MODEL_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
        GROUNDING_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "groundingdino_swint_ogc.pth")
        if not os.path.exists(GROUNDING_MODEL_PATH):
            print("Downloading GroundingDino model...")
            urlretrieve(GROUNDING_MODEL_URL, GROUNDING_MODEL_PATH)
            print(f"Model saved to {GROUNDING_MODEL_PATH}")
        
        self.grounding_model = load_model('./src/groundingdino/config/GroundingDINO_SwinT_OGC.py', GROUNDING_MODEL_PATH,"cuda")
        # import pdb
        # pdb.set_trace()
        # self.grounding_model = load_model('./src/eval/groundingdino/config/GroundingDINO_SwinT_OGC.py', './src/eval/submission/checkpoint/groundingdino_swint_ogc.pth',"cuda")


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

        # self.stats = pickle.load(open("./src/eval/submission/memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl", "rb"))
        self.stats = pickle.load(open("./src/eval/submission/memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl", "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False  # True #False
        self.text_prompt = {'breakfast_box': {'box_threshold': 0.25, 'text_threshold': 0.2, 'text_prompt': "almond . apple . container . oatmeal . orange . banana . ", 'background_prompt': ""}, 'juice_bottle': {'box_threshold': 0.3, 'text_threshold': 0.3, 'text_prompt': "alcohol. apple juice. beverage. bottle. liquor. glass bottle. juice. lemonade. liquid. olive oil. orange juice. yellow banana. bottle. glass bottle. glass jar. jug. juice. liquid. milk beverage. bottle. cherry. condiment. liquor. glass bottle. honey. juice. liquid. maple syrup. sauce. syrup. tomato sauce.", 'background_prompt': ""}, 'pushpins': {'box_threshold': 0.3, 'text_threshold': 0.2, 'text_prompt': "pin .", 'background_prompt': "container ."}, 'screw_bag': {'box_threshold': 0.3, 'text_threshold': 0.3, 'text_prompt': "metal bolts. metal hex nuts. metal washers. metal screws. zip-lock bag. screw. ring. metal circle. bag. bolt. container. nut. package. plastic. screw. tool.", 'background_prompt': "zip-lock bag."}, 'splicing_connectors': {'box_threshold': 0.3, 'text_threshold': 0.2, 'text_prompt': "attach. cable. connector. hook. electric outlet. plug. pole. socket. wire.", 'background_prompt': ""}}


    def set_viz(self, viz):
        self.visualization = viz

    def set_val(self, val):
        self.validation = val

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
                    label_map[label_a] = background_class  # default background

            merged_a = label_map[a]
            return merged_a
        
        def remove_overlapping_boxes(boxes, thr=0.5):
            """
            通过计算交并比(IoU)去除重叠的矩形框
            
            参数:
                boxes: 矩形框列表，每个矩形框格式为[xmin, ymin, xmax, ymax]
                thr: IoU阈值，大于此阈值则认为重叠(默认0.5)
            
            返回:
                去除重叠后的矩形框列表
            """
            if not boxes:
                return []
            
            # 将boxes转换为float类型并拷贝，避免修改原始数据
            boxes = [list(map(float, box)) for box in boxes]
            
            # 按矩形面积从大到小排序
            boxes.sort(key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
            
            keep = []  # 保留的矩形框
            
            while boxes:
                current = boxes.pop(0)  # 取出当前最大的框
                keep.append(current)
                
                # 计算剩余框与当前框的IoU，并筛选出IoU小于阈值的框
                boxes = [box for box in boxes if iou_box(current, box) <= thr]
            
            return keep

        def iou_box(box1, box2):
            # 计算交集区域
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # 计算交集面积
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            
            # 计算各自面积
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # 计算并集面积
            union_area = area1 + area2 - inter_area
            
            # 计算IoU
            return inter_area / union_area if union_area > 0 else 0

        pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device)
        kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)  # default to background

        
        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        time1 = time.time()
        # masks = self.mask_generator.generate(raw_image)
        
        # 维护数据库
        '''
        self.db = TensorDictDB(self.class_name)
        result = self.db.get(raw_image)
        if result:
            print('***Exist in h5!')
            masks = result
        else:
        '''
        if True:
            # masks = self.mask_generator.generate(raw_image)
            
            ####
            # groundingSAM_start
            sam = self.model_sam
            predictor = SamPredictor(sam)
            text_prompt = self.text_prompt[self.class_name]['text_prompt']
            background_prompt = self.text_prompt[self.class_name]['background_prompt']
            box_threshold = self.text_prompt[self.class_name]['box_threshold']
            text_threshold = self.text_prompt[self.class_name]['text_threshold']
            boxes_filt, pred_phrases = get_grounding_output(
                self.grounding_model, image[0], text_prompt, box_threshold, text_threshold, device="cuda"
            )

            background_box = list()
            for i,text in enumerate(pred_phrases):
                for j in background_prompt.split('.'):
                    if j in text.replace(' - ','-') and j != ' ' and j != '':
                        background_box.append(i)

            predictor.set_image(raw_image)

            H, W = 448, 448
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu() # (xmin, ymin, xmax, ymax)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            ###
            # 去重重叠框
            boxes = boxes_filt.cpu().numpy()
            boxes_nofiltered = []
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = map(int, box)
                boxes_nofiltered.append([xmin, ymin, xmax, ymax])
            
            boxes_filtered = remove_overlapping_boxes(boxes_nofiltered, thr=0.3)
            ###

            masks0, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )

            masks = []
            for mask in masks0:
                mask = torch.squeeze(mask, dim=0)
                # resized_tensor = F.interpolate(
                #     mask.unsqueeze(0).unsqueeze(0),  # 增加 batch 和 channel 维度 (1, 1, 1280, 1600)
                #     size=(448, 448),
                #     mode='bilinear',  # 或 'nearest', 'bicubic'
                # ).squeeze(0).squeeze(0)  # 恢复原始维度 (448, 448)
                # mask = resized_tensor.cpu().numpy()
                mask = mask.cpu().numpy()
                masks.append({'segmentation': mask, 'area': np.sum(mask)})
            # groundingSAM_end
            ####
            
            # self.db.add(raw_image, masks)
        time2 = time.time()
        print("sam time:", time2 - time1)

        ####
        # cal pushpins num
        sam = self.model_sam
        predictor = SamPredictor(sam)
        text_prompt = self.text_prompt[self.class_name]['text_prompt']
        background_prompt = self.text_prompt[self.class_name]['background_prompt']
        box_threshold = self.text_prompt[self.class_name]['box_threshold']
        text_threshold = self.text_prompt[self.class_name]['text_threshold']
        boxes_filt, pred_phrases = get_grounding_output(
            self.grounding_model, image[0], text_prompt, box_threshold, text_threshold, device="cuda"
        )

        background_box = list()
        for i,text in enumerate(pred_phrases):
            for j in background_prompt.split('.'):
                if j in text.replace(' - ','-') and j != ' ' and j != '':
                    background_box.append(i)

        predictor.set_image(raw_image)

        H, W = 448, 448
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu() # (xmin, ymin, xmax, ymax)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        ###
        # 去重重叠框
        boxes = boxes_filt.cpu().numpy()
        boxes_nofiltered = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            boxes_nofiltered.append([xmin, ymin, xmax, ymax])
        
        boxes_filtered = remove_overlapping_boxes(boxes_nofiltered, thr=0.3)
        #### 
        

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
        if self.class_name == 'pushpins':
            # object count hist
            kernel = np.ones((3, 3), dtype=np.uint8)  # dilate for robustness
            binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(
                np.uint8)  # foreground 1  background 0
            dilate_binary = cv2.dilate(binary, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            pushpins_count = num_labels - 1  # number of pushpins

            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),
                                           interpolation=cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))


            pushpins_count = len(boxes_filtered)
            if self.few_shot_inited and pushpins_count != self.pushpins_count and self.anomaly_flag is False:
                self.anomaly_flag = True
                print('number of pushpins: {}, but canonical number of pushpins: {}'.format(pushpins_count,
                                                                                            self.pushpins_count))

            # patch hist
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            binary_foreground = dilate_binary.astype(np.uint8)

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

            # todo: same number in total but in different boxes or broken box
            return {"score": score, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'splicing_connectors':
            #  object count hist for default
            sam_mask_max_area = sorted_masks[0]['segmentation']  # background
            binary = (sam_mask_max_area == 0).astype(
                np.uint8)  # sam_mask_max_area is background,  background 0 foreground 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:  # 448x448 64
                    binary[temp_mask] = 0  # set to background
                else:
                    count += 1
            if count != 1 and self.anomaly_flag is False:  # cable cut or no cable or no connector
                print(
                    'number of connected component in splicing_connectors: {}, but the default connected component is 1.'.format(
                        count))
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

            slice_cable = binary[:, int(w / 2) - 1: int(w / 2) + 1]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
            cable_count = num_labels - 1
            if cable_count != 1 and self.anomaly_flag is False:  # two cables
                print('number of cable count in splicing_connectors: {}, but the default cable count is 1.'.format(
                    cable_count))
                self.anomaly_flag = True

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

    def process_k_shot(self, class_name, few_shot_samples, few_shot_paths):
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
        if len(patch_token_hist) != 0:  # patch hist
            self.patch_token_hist = np.stack(patch_token_hist)
        if len(mem_instance_masks) != 0:
            self.mem_instance_masks = mem_instance_masks

        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset


    def process(self, class_name: str, few_shot_samples: list[torch.Tensor], few_shot_paths: list[str]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device)
        scores, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(
            class_name, few_shot_samples, few_shot_paths)

    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        few_shot_samples = setup_data.get("few_shot_samples")
        class_name = setup_data.get("dataset_category")
        few_shot_paths = setup_data.get("few_shot_samples_path")
        self.class_name = class_name

        self.k_shot = few_shot_samples.size(0)
        self.process(class_name, few_shot_samples, few_shot_paths)
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


    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
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

        self.anomaly_flag = False
        batch = image
        batch = self.transform(batch).to(self.device)
        results = self.forward_one_sample(batch, self.mem_patch_feature_clip_coreset,
                                          self.mem_patch_feature_dinov2_coreset)

        hist_score = results['hist_score']
        structural_score = results['structural_score']
        instance_hungarian_match_score = results['instance_hungarian_match_score']

        anomaly_map_structural = results['anomaly_map_structural']

        if self.validation:
            return {"hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score),
                    "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # standardization

        standard_structural_score = (structural_score - self.stats[self.class_name]["structural_scores"]["mean"]) / \
                                    self.stats[self.class_name]["structural_scores"]["unbiased_std"]
        standard_instance_hungarian_match_score = (instance_hungarian_match_score -
                                                   self.stats[self.class_name]["instance_hungarian_match_scores"][
                                                       "mean"]) / \
                                                  self.stats[self.class_name]["instance_hungarian_match_scores"][
                                                      "unbiased_std"]

        '''
        # ori
        pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
        
        standard_hist_score = (hist_score - self.stats[self.class_name]["hist_scores"]["mean"])/self.stats[self.class_name]["hist_scores"]["unbiased_std"]
        pred_score = sigmoid(pred_score)
        '''
        standard_hist_score = (hist_score - self.stats[self.class_name]["hist_scores"]["mean"]) / self.stats[self.class_name]["hist_scores"]["unbiased_std"]
        if self.class_name == 'breakfast_box':
            a1 = 16/(16+4+19)
            a2 = 4/(16+4+19)
            a3 = 19/(16+4+19)
            pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
           
        elif self.class_name == 'juice_bottle':
            a1 = 2/(2+18+1) 
            a2 = 18/(2+18+1)
            a3 = 1/(2+18+1)
            pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
        elif self.class_name == 'screw_bag':
            a1 = 1/(1+2+16)
            a2 = 2/(1+2+16)
            a3 = 16/(1+2+16)
            pred_score = a1*standard_structural_score + a2*standard_instance_hungarian_match_score + a3*standard_hist_score
        else:
            pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
        pred_score = sigmoid(pred_score)
        if self.anomaly_flag:
            pred_score = 1.
            self.anomaly_flag = False

        #return {"pred_score": torch.tensor(pred_score), "anomaly_map": torch.tensor(anomaly_map_structural),
        #        "hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score),
        #        "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}
        batch_size = image.shape[0]
        pred_score = torch.tensor(pred_score).to(self.device)
        return ImageBatch(image=image, pred_score=pred_score,)



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
