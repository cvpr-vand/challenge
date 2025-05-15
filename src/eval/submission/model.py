"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn
from torchvision.transforms import v2

#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
from eval.submission.utils.prompt_ensemble import encode_text_with_prompt_ensemble, encode_normal_text, encode_abnormal_text, encode_general_text, encode_obj_text, encode_obj_text2
from scipy.optimize import linear_sum_assignment
import time
import sys
sys.path.append('/home/user/actions-runner/_work/challenge/challenge/src/eval/submission/gdino_sam2')
from eval.submission.gdino_sam2.interface import GSAM2Predictor
import os




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

        #setup_seed(42)
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

        #self.model_sam = sam_model_registry["vit_h"](checkpoint="./submission/checkpoint/sam_vit_h_4b8939.pth").to(self.device)
        #self.model_sam = sam_model_registry["vit_b"](checkpoint = "./submission/checkpoint/sam_vit_b_01ec64.pth").to(self.device)
        #self.mask_generator = SamAutomaticMaskGenerator(model=self.model_sam)
        self.mask_generator = GSAM2Predictor()

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
            "breakfast_box": [1, 2, 3, 4, 5],
            "juice_bottle": [1,2,3],
            "pushpins": [1,2],
            "screw_bag": [1,2,3],
            "splicing_connectors": [1, 2]
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

        from submission.dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        self.stats = pickle.load(open("./submission/memory_bank/statistic_scores_model_ensemble_few_shot_val.pkl", "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False  # True #False

        self.objs_dict = {
            # 5 大类别
            "breakfast_box": [["white box"], ["orange",  "orange peach"], ["peach",], ["oatmeal"], ["banana chips", "almonds", "banana chips almonds "]],
            # 5 大类别
            "juice_bottle": [["glass bottle"], ["cherry", "orange", "banana"], ["label", "tag", "label tag"]],
            # 两个类别
            "pushpins": [["tools"], ["pushpins"],["fine stick"]],
            # 3个类别
            "screw_bag": [["plastic bag"], ["metal circle", "circle"], ["long bolts", "bolt", "not circle"]],
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
        self.mask_generator.objs_dict = self.objs_dict
        self.mask_generator.foreground_num = self.foreground_num


    def set_viz(self, viz):
        self.visualization = viz

    def set_val(self, val):
        self.validation = val

    def generate_mask(self, sample):
        masks, boxes, obj_names, pil_img = self.mask_generator.forward(sample, self.class_name)     # SAM mask
        clusted_masks, label_ids = self.mask_generator.cluster_mask(masks, boxes, obj_names, self.class_name)
        return clusted_masks, boxes, label_ids, pil_img

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

        pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device)
        kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)  # default to background

        
        #raw_image = to_np_img(image[0])
        #height, width = raw_image.shape[:2]
        time1 = time.time()
        #masks = self.mask_generator.generate(raw_image)
        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        masks, boxes, label_ids, pil_img = self.generate_mask(raw_image)
        if class_name == "pushpins":
            masks[0][0:35, 200:250] = masks[1][0:35, 200:250]
            masks[1][0:35, 200:250] = 0
            label2 = masks[1]  ###pushpins
            label1 = masks[0]  ###box

            ####
            contours, _ = cv2.findContours(label1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_contour = np.zeros((label1.shape[0], label1.shape[1]))
            image_contour = cv2.drawContours(image_contour, contours, -1, 1, -1)
            image_contour[label2==1] = 0
            masks[0] = image_contour


        time2 = time.time()
        print("sam time:", time2 - time1)
        ###save mask
        for cat_id, mask in enumerate(masks):
            path_rgb = path + "_ori.jpg"
            pil_img.save(path_rgb)
            path_save = path + '_' + str(cat_id) +".jpg"
            cv2.imwrite(path_save, mask * 255)

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

        #raw_image = to_np_img(image[0])
        #height, width = raw_image.shape[:2]
        #time1 = time.time()
        #masks = self.mask_generator.generate(raw_image)
        #time2 = time.time()
        #print("sam time:", time2 - time1)
        

        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy()
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1)
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        #sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        #sam_mask = plot_results_only(sorted_masks).astype(np.int32)
        sam_mask = np.zeros((masks[0].shape[0], masks[0].shape[1]))
        cur = 1
        for mask in masks:
            m = mask==1
            sam_mask[m] = cur
            cur += 1
    

        #resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        #merge_sam = merge_segmentations(sam_mask, resized_mask, background_class=self.classes - 1)
        merge_sam = sam_mask

        #resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        #patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask,background_class=self.patch_query_obj.shape[0] - 1)
        patch_merge_sam = sam_mask.astype(np.int64)

        # filter small region for merge sam
        binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32:  # 448x448
                #merge_sam[temp_mask] = self.classes - 1  # set to background
                merge_sam[temp_mask] = 0

        # filter small region for patch merge sam
        #binary = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)  # foreground 1  background 0
        binary = np.isin(patch_merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)   # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32:  # 448x448
                #patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0] - 1  # set to background
                patch_merge_sam[temp_mask] = 0

        score = 0.  # default to normal
        self.anomaly_flag = False
        instance_masks = []
        if self.class_name == 'pushpins':
            # object count hist
            #kernel = np.ones((3, 3), dtype=np.uint8)  # dilate for robustness
            #binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
            #dilate_binary = cv2.dilate(binary, kernel)
            #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            #pushpins_count = num_labels - 1  # number of pushpins

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masks[1], connectivity=8)
            pushpins_count = num_labels - 1  # number of pushpins

            '''
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size),interpolation=cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))
            '''
            for mask in masks:
                mask = cv2.resize(mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
                if mask.any():
                    instance_masks.append(mask.astype(np.bool_).reshape(-1))


            if self.few_shot_inited and pushpins_count != self.pushpins_count and self.anomaly_flag is False:
                self.anomaly_flag = True
                print('number of pushpins: {}, but canonical number of pushpins: {}'.format(pushpins_count,
                                                                                            self.pushpins_count))

            # patch hist
            clip_patch_hist = None
            if not self.few_shot_inited:
                clip_patch_hist = np.bincount(sam_mask.astype(np.int64).reshape(-1), minlength=self.patch_query_obj.shape[0])
                clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited and self.anomaly_flag is False:
                clip_patch_hist = np.bincount(sam_mask.astype(np.int64).reshape(-1),minlength=self.patch_query_obj.shape[0])
                clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            #binary_foreground = dilate_binary.astype(np.uint8)

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]


            # todo: same number in total but in different boxes or broken box
            return {"score": score, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'splicing_connectors':
            #  object count hist for default
            #sam_mask_max_area = sorted_masks[0]['segmentation']  # background
            binary = (sam_mask != 0).astype(np.uint8)  # sam_mask_max_area is background,  background 0 foreground 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:  # 448x448 64
                    binary[temp_mask] = 0  # set to background
                else:
                    count += 1
            if count != 1 and self.anomaly_flag is False:  # cable cut or no cable or no connector
                print('number of connected component in splicing_connectors: {}, but the default connected component is 1.'.format( count))
                self.anomaly_flag = True
            if np.sum(binary) >= 448 * 448 * 0.8 and self.anomaly_flag is False:
                self.anomaly_flag = True

            h, w = binary.shape
            left, right = binary[:, :int(w / 2)], binary[:, int(w / 2):]
            if ((np.sum(left) == 0) or (np.sum(right) == 0)) and self.anomaly_flag is False:
                self.anomaly_flag = True
            if self.anomaly_flag is False and self.few_shot_inited:
                if ((np.sum(left) == 0) or (np.sum(right) == 0)):
                    self.anomaly_flag = True
                else:
                    contours_left, _ = cv2.findContours(left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_right, _ = cv2.findContours(right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_left = sorted(contours_left, key=cv2.contourArea, reverse=True)
                    largecontour_left = contours_left[0]
                    area_left= cv2.contourArea(largecontour_left)
                    contours_right = sorted(contours_right, key=cv2.contourArea, reverse=True)
                    largecontour_right = contours_right[0]
                    area_right = cv2.contourArea(largecontour_right)
                    ratio = area_left / area_right
                    if ratio < 0.6 or ratio > 1.4:  # too short or too long centroids distance (cable) # 0.6 1.4
                        self.anomaly_flag = True

            if  self.anomaly_flag is False and self.few_shot_inited:
                mask_tmp = masks[0].copy()
                sum_col = np.sum(mask_tmp*255, axis=0)
                sum_col = sum_col.astype(np.int64)
                i_list = [i for i in range(binary.shape[0])]
                markbegin = 0
            
                

                
                for i in range(10, 224):
                    if sum_col[i + 10] - sum_col[i] < -15000:
                        markbegin = i + 10
                        break
                markend = 224
                for i in range(224, 400):
                    if abs(sum_col[i + 10] - sum_col[i]) > 15000:
                        markend = i
                        break
                
                cable = np.zeros((448, 448))
                cable[:, markbegin: markend] = binary[:, markbegin:markend]
                kernel = np.ones((3, 3), dtype=np.uint8)
                cable = cv2.erode(cable, kernel)
                cable1 = cv2.dilate(cable, kernel)
                block = binary - cable1
                block = cv2.erode(block, kernel)
                markbegin_y = 0
                for i in range(448):
                    if cable[i, markbegin + 2] != 0:
                        markbegin_y = i
                        break
                markend_y = 0
                for i in range(448):
                    if cable[i, markend - 2] != 0:
                        markend_y = i
                        break
                markbeginblock_y = 0
                for i in range(448):
                    if block[i, markbegin - 10] != 0:
                        markbeginblock_y = i
                        break
                markendblock_y = 0
                for i in range(448):
                    if block[i, markend + 10] != 0:
                        markendblock_y = i
                        break
                
            
            
                difference = abs(abs(markbeginblock_y - markbegin_y) - abs(markendblock_y - markend_y))
                if difference>100 :  # too short or too long centroids distance (cable) # 0.6 1.4
                    self.anomaly_flag = True



            foreground_pixel_count = 0
            distance = 0
            sam_patch_hist = 0

            '''
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
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
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
                print('number of cable count in splicing_connectors: {}, but the default cable count is 1.'.format( cable_count))
                self.anomaly_flag = True

            # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:  # color and number mismatch
                    print(
                        'cable color and number of clamps mismatch, cable color idx: {} (0: yellow 2-clamp, 1: blue 3-clamp, 2: red 5-clamp), foreground_pixel_count :{}, canonical foreground_pixel_hist: {}.'.format(idx_color, foreground_pixel_count, self.foreground_pixel_hist))
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
            '''

            # patch hist
            patch_merge_sam = masks[0].astype(np.int64)

            if not self.few_shot_inited:
                sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])  # [:-1]  # ignore background (grid) for statistic
                sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited and self.anomaly_flag is False:
                sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])  # [:-1]  # ignore background (grid) for statistic
                sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()
                print("hisscore:", score)
                if ratio < 0.85 or ratio > 1.15:
                    score = 1.0
                else:
                    score = -1.0


            # todo    mismatch cable link
            '''
            #binary_foreground = binary.astype(np.uint8)  # only 1 instance, so additionally seperate cable and clamps
            if binary_connector.any():
                instance_masks.append(binary_connector.astype(np.bool_).reshape(-1))
            if binary_clamps.any():
                instance_masks.append(binary_clamps.astype(np.bool_).reshape(-1))
            if binary_cable.any():
                instance_masks.append(binary_cable.astype(np.bool_).reshape(-1))
            '''
            mask = cv2.resize(masks[0], (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
            if mask.any():
                instance_masks.append(mask.astype(np.bool_).reshape(-1))
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "distance": distance,
                    "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'screw_bag':
            foreground_pixel_count = 0
            # patch hist
            clip_patch_hist = np.bincount(sam_mask.astype(np.int64).reshape(-1), minlength=self.patch_query_obj.shape[0])[:-1]
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # # todo: count of screw, nut and washer, screw of different length
            for mask in masks:
                mask = cv2.resize(mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
                if mask.any():
                    instance_masks.append(mask.astype(np.bool_).reshape(-1))


            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

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
            for mask in masks:
                mask = cv2.resize(mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
                if mask.any():
                    instance_masks.append(mask.astype(np.bool_).reshape(-1))



            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

            return {"score": score, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'juice_bottle':
            # remove noise due to non sam mask
            #merge_sam[sam_mask == 0] = self.classes - 1
            #patch_merge_sam[sam_mask == 0] = self.patch_query_obj.shape[0] - 1  # 79.5

            # [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'], ['black background', 'background']],
            # fruit and liquid mismatch (todo if exist)
            resized_patch_merge_sam = cv2.resize(patch_merge_sam, (self.feat_size, self.feat_size),interpolation=cv2.INTER_NEAREST)
            binary_liquid = (resized_patch_merge_sam == 1)
            binary_fruit = (resized_patch_merge_sam == 2)
            
    
            #query_liquid = encode_obj_text(self.model_clip, self.juice_bottle_liquid_query_words_dict, self.tokenizer,self.device)
            #query_fruit = encode_obj_text(self.model_clip, self.juice_bottle_fruit_query_words_dict, self.tokenizer,self.device)
        
            
            #liquid_feature = proj_patch_token[binary_liquid.reshape(-1), :].mean(0, keepdim=True)
            #liquid_idx = (liquid_feature @ query_liquid.T).argmax(-1).squeeze(0).item()

            #fruit_feature = proj_patch_token[binary_fruit.reshape(-1), :].mean(0, keepdim=True)
            #fruit_idx = (fruit_feature @ query_fruit.T).argmax(-1).squeeze(0).item()

            #if (liquid_idx != fruit_idx) and self.anomaly_flag is False or (self.anomaly_flag is False and len(masks) < 3):
            if  (self.anomaly_flag is False and len(masks) < 3):
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

            for mask in masks:
                mask = cv2.resize(mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
                if mask.any():
                    instance_masks.append(mask.astype(np.bool_).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)  # [N, 64x64]

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


        for image, cluster_feature, proj_patch_token, few_shot_path in zip(few_shot_samples.chunk(self.k_shot),
                                                                           cluster_features.chunk(self.k_shot),
                                                                           proj_patch_tokens.chunk(self.k_shot),
                                                                           few_shot_paths):
            # path = os.path.dirname(few_shot_path).split('/')[-1] + "_" + os.path.basename(few_shot_path).split('.')[0]
            self.anomaly_flag = False
            path = "few_shot_" + os.path.basename(few_shot_path).split('.')[0]
            dir_path = os.path.join('./train/',class_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            image_path = dir_path + '/' + path
            import pdb
            pdb.set_trace()
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name, image_path)
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
                           mem_patch_feature_dinov2_coreset: torch.Tensor, batch_path):

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


        if "good" in batch_path[0]:    
            path = "few_shot_good_" + os.path.basename(batch_path[0]).split('.')[0]
        if "logical" in batch_path[0]:
            path = "few_shot_logical_" + os.path.basename(batch_path[0]).split('.')[0]
        if "structural" in batch_path[0]:
            path = "few_shot_structural_" + os.path.basename(batch_path[0]).split('.')[0]
        dir_path = os.path.join('./test/', self.class_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        image_path = dir_path + '/' + path

        time1 = time.time()
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name, image_path)
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


    def forward(self, image: torch.Tensor, batch_path) -> dict[str, torch.Tensor]:
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
        #import pdb
        #pdb.set_trace()
        print("batch_path:", batch_path)
        self.anomaly_flag = False
        batch = image
        batch = self.transform(batch).to(self.device)
        results = self.forward_one_sample(batch, self.mem_patch_feature_clip_coreset,
                                          self.mem_patch_feature_dinov2_coreset, batch_path)

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
        standard_instance_hungarian_match_score = (instance_hungarian_match_score -self.stats[self.class_name]["instance_hungarian_match_scores"][ "mean"]) / \
                                                  self.stats[self.class_name]["instance_hungarian_match_scores"][ "unbiased_std"]
        print("hist_score,structural_score, instance_hungarian_match_score:", hist_score,structural_score, instance_hungarian_match_score)
        print("hist_score,structural_score, instance_hungarian_match_score:", hist_score, standard_structural_score, standard_instance_hungarian_match_score)
        pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
        if self.class_name == "breakfast_box":
            pred_score = standard_structural_score
        if self.class_name == "splicing_connectors":
             pred_score = hist_score + standard_structural_score * 0.6 + standard_instance_hungarian_match_score *0.4
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

