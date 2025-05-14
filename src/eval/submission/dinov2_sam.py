import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torchvision.transforms import v2
import torch.nn.functional as F
import numpy as np
import cv2
from .gdino_sam2 import GSAM2Predictor
from .sampler import GreedyCoresetSampler, ApproximateGreedyCoresetSampler
from timm import create_model
import csv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]

'''
save_results_to_csv(image_path, 
                    anomaly_map_ret_dinov2.max().item(), 
                    anomaly_map_ret_eva02.max().item(),
                    dinov2_anomaly_map_ret_part.max().item(),
                    dinov2_subcategory_mean_dists.max().item(),
                    eva02_anomaly_map_ret_part.max().item(),
                    eva02_subcategory_mean_dists.max().item(),
                    area_hist_avg, color_hist_avg, output_path)
'''
def save_results_to_csv(image_path, 
                        anomaly_map_ret_dinov2, 
                        anomaly_map_ret_eva02, 
                        dinov2_anomaly_map_ret_part, 
                        dinov2_subcategory_mean_dists,
                        eva02_anomaly_map_ret_part,
                        eva02_subcategory_mean_dists,
                        area_hist_avg, color_hist_avg,
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


class AnomalyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.image_size = 224
        device = torch.device("cuda")
        self.out_layers = [5, 11, 17, 23]
        self.device = device
        self.embedding_dim = 1024
        self.orginal_grid_size = 16
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
        eva02_pretrained_cfg = {"file": "models/eva02_large_patch14_224.mim_m38m/model.safetensors"}
        self.model_eva02 = create_model('eva02_large_patch14_224.mim_m38m', pretrained=True, pretrained_cfg=eva02_pretrained_cfg, num_classes=0, img_size=self.image_size).to(self.device)
        self.model_eva02.eval()
        dinov2_pretrained_cfg = {"file": "models/vit_large_patch14_dinov2.lvd142m/model.safetensors"}
        self.model_dinov2 = create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, pretrained_cfg=dinov2_pretrained_cfg, num_classes=0, img_size=self.image_size).to(self.device)

        self.model_dinov2.eval()
        self.gdino_init = False
        self.vis = False
        self.class_hists_train = None
        self.class_color_train = None
        self.pushpins_count = 15
        self.splicing_connectors_count = [2, 3, 5] # coresponding to yellow, blue, and red

        self.mask_generator = GSAM2Predictor()
        self.gdino_init = True

    def compute_patch_similarity(self, query_feats, gallery_feats, feat_dim):
        """通用patch特征相似度计算函数
        Args:
            query_feats: 待查询特征 [N, D]
            gallery_feats: 比对特征 [M, D] 
            feat_dim: 特征维度
        Returns:
            相似度得分 [N]
        """
        # 维度检查
        assert query_feats.shape[-1] == feat_dim, f"特征维度应为{feat_dim}"
        assert gallery_feats.shape[-1] == feat_dim, f"特征维度应为{feat_dim}"

        # 重塑特征维度 [N,1,D] 和 [1,M,D]
        query = query_feats.view(-1, 1, feat_dim)
        gallery = gallery_feats.view(1, -1, feat_dim)
        
        # 批量计算余弦相似度 [N,M]
        sim_matrix = F.cosine_similarity(query, gallery, dim=2)
        
        # 获取每个query与gallery的最大相似度 [N]
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
        # ================== DINOv2特征处理 ==================
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
        

        # 合并各层结果并平均
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
        
        for layer in range(num_layers):
            subcategory_mean_dists_layer = []
            subcategory_patch_sims_layer = []
            
            for cat_id in range(self.foreground_num[self.class_name]):
                # 处理子类别特征和补丁
                subcategory_test_feat, subcategory_test_patch = self.process_subcategory(
                    cat_id, test_patch_tokens[layer], heatmap_cache
                )
                
                if len(subcategory_test_feat) == 0: 
                    continue
                if len(normal_feats[layer][cat_id]) == 0:
                    continue
                # 计算特征距离
                # print(normal_feats[layer][cat_id])
                dist = 1 - (subcategory_test_feat @ normal_feats[layer][cat_id].T).max().item()
                subcategory_mean_dists_layer.append(dist)
                
                # 计算补丁相似度
                subcategory_test_patch_reshaped = subcategory_test_patch.reshape(-1, 1, self.embedding_dim)
                subcategory_normal_patch_reshaped = normal_patch_tokens[layer][cat_id].reshape(1, -1, self.embedding_dim)
                
                cosine_similarity_matrix = F.cosine_similarity(
                    subcategory_test_patch_reshaped, 
                    subcategory_normal_patch_reshaped, 
                    dim=2
                )
                
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

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.pushpins_abnormal_flag = 0
        self.connector_abnormal_flag = 0
        self.bottle_abnormal_flag = 0
        image_path = batch["image_path"]
        batch = batch["image"]
        # if self.class_name == "splicing_connectors":
        #    batch = v2.functional.crop(batch, 15, 15, 226, 226)
        heatmap_cache = dict()
        area_hist = list()
        color_hist = list()
        if self.gdino_init:
            clusted_masks, boxes, label_ids, pil_img = self.generate_mask(batch[0])
            if self.class_name == "pushpins":
                if len(label_ids[-1]) != self.pushpins_count:
                        self.pushpins_abnormal_flag = 1
            if self.class_name == "juice_bottle":
                if len(label_ids[-1]) != 2:
                    self.bottle_abnormal_flag = 1
            if self.class_name == "splicing_connectors":
                if len(clusted_masks) > 0:
                    # 长度限制
                    box = boxes[label_ids[0][0]]
                    connector_width_ratio = (box[2] - box[0]) / pil_img.width
                    if connector_width_ratio < 0.7:
                        # print(image_path,  "连接器宽度过窄")
                        self.connector_abnormal_flag = 1
                    # 直线问题
                    binary = clusted_masks[0]
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
                    if num_labels > 2 and np.sum(labels == 1) > 5000 and np.sum(labels == 2) > 5000: # 直线断开
                        # print(image_path,  "直线断开")
                        self.connector_abnormal_flag = 1
                    for i in range(1, num_labels):
                        temp_mask = labels == i
                        if np.sum(temp_mask) <= 64: # 448x448 64
                            binary[temp_mask] = 0 # set to background
                    # 分割直线和连接器
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    rect_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                    # 使用连通域分析统计矩形个数
                    num_labels_rects, labels_rects, stats_rects, _ = cv2.connectedComponentsWithStats(rect_mask, connectivity=8)
                    if num_labels_rects != 3:
                        # print(f"{image_path} 连接器个数不对")
                        self.connector_abnormal_flag = 1
                    else:
                        rectangle_areas = [stats_rects[i, cv2.CC_STAT_AREA] for i in range(1, num_labels_rects)]
                        if len(rectangle_areas) > 1:
                            ratio = rectangle_areas[0] / (rectangle_areas[1] + 1e-5)
                            if (ratio > 1.2 or ratio < 0.8): 
                                self.connector_abnormal_flag = 1
                    only_line = cv2.subtract(binary, rect_mask)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(only_line, connectivity=8)
                    min_area = 1000  # 面积阈值
                    final_line = np.zeros_like(only_line)
                    for i in range(1, num_labels):  # 跳过背景
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            final_line[labels == i] = 255
                    num_labels_line, labels_line, stats_line, _ = cv2.connectedComponentsWithStats(final_line, connectivity=8)
                    if num_labels_line != 2:
                        self.connector_abnormal_flag = 1
                    # 颜色匹配
                    # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch

                    

                # 连接器左右个数是否匹配
                if len(label_ids) == 2 and len(label_ids[-1]) == 2:
                    box1 = boxes[label_ids[-1][0]]
                    box2 = boxes[label_ids[-1][1]]
                    box1_height = abs(box1[3] - box1[1])
                    box2_height = abs(box2[3] - box2[1])
                    if abs(box1_height - box2_height)> 0.06 * pil_img.height:
                        # print(image_path,  "连接器高度不一致")
                        self.connector_abnormal_flag = 1
                

            # 转换为LAB色彩空间
            imglabo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2LAB)
            color_a = imglabo[:, :, 1].astype(np.float32)  # 使用float32提高精度
            color_b = imglabo[:, :, 2].astype(np.float32)
            epsilon = 1e-7  # 更小的除数保护值
            color_ratio_sq = (color_b / (color_a + epsilon)) ** 2
            for cat_id, mask in enumerate(clusted_masks):
                heatmap_cache[f"./outputs/{self.class_name}/test/000/mask_{cat_id}.png"] = mask
                area = mask.sum()
                area_hist.append(area)
                masked_color = color_ratio_sq * mask
                color_value = np.sum(masked_color) / (area + epsilon)
                color_hist.append(color_value)
                if self.vis:
                    path = f"./outputs/{self.class_name}/test/000/img.png"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    pil_img.save(path)
                    for cat_id, mask in enumerate(clusted_masks):
                        path = f"./outputs/{self.class_name}/test/000/mask_{cat_id}.png"
                        
                        cv2.imwrite(path, mask*255) 
        area_hist_avg = np.array(area_hist) / self.class_hists_train
        color_hist_avg = np.array(color_hist) / self.class_color_train
        transformed_image = self.transform_dino(batch)

        test_patch_tokens_dinov2 = self.model_dinov2.forward_intermediates(transformed_image, indices=self.out_layers, intermediates_only=True)
        test_patch_tokens_dinov2 = [p.permute(0, 2, 3, 1).view(-1, self.orginal_grid_size**2, self.embedding_dim) for p in test_patch_tokens_dinov2]
        anomaly_map_ret_dinov2 = self.compute_anomaly_maps(test_patch_tokens_dinov2, self.dinov2_normal_patch_tokens)
        
        test_patch_tokens_eva02 = self.model_eva02.forward_intermediates(transformed_image, indices=self.out_layers, intermediates_only=True)
        test_patch_tokens_eva02 = [p.permute(0, 2, 3, 1).view(-1, self.orginal_grid_size**2, self.embedding_dim) for p in test_patch_tokens_eva02]
        anomaly_map_ret_eva02 = self.compute_anomaly_maps(test_patch_tokens_eva02, self.eva02_normal_patch_tokens)

        dinov2_anomaly_map_ret_part, dinov2_subcategory_mean_dists = self.calculate_subcategory_metrics(
            test_patch_tokens_dinov2, heatmap_cache, self.selected_features_dinov2, self.part_normal_patch_tokens_dinov2
        )
        eva02_anomaly_map_ret_part, eva02_subcategory_mean_dists = self.calculate_subcategory_metrics(
            test_patch_tokens_eva02, heatmap_cache, self.selected_features_eva02, self.part_normal_patch_tokens_eva02
        )
        output_path = "data.csv"
        # save_results_to_csv(image_path, 
        #                     anomaly_map_ret_dinov2.max().item(), 
        #                     anomaly_map_ret_eva02.max().item(),
        #                     dinov2_anomaly_map_ret_part.max().item(),
        #                     dinov2_subcategory_mean_dists.max().item(),
        #                     eva02_anomaly_map_ret_part.max().item(),
        #                     eva02_subcategory_mean_dists.max().item(),
        #                     area_hist_avg, color_hist_avg, output_path)
        return {
            "pred_score": torch.tensor( anomaly_map_ret_dinov2.max().item())
        }
        # if self.class_name == "juice_bottle":
        #     return {
        #         "pred_score": torch.tensor( 
        #             20 * anomaly_map_ret_dinov2.max().item() +
        #             4 * anomaly_map_ret_eva02.max().item() +  
        #             4 *   dinov2_anomaly_map_ret_part.max().item() +
        #             8 * eva02_anomaly_map_ret_part.max().item() +
        #             4 * abs(area_hist_avg[-1] - 1) + 
        #             4 * abs(area_hist_avg[0] - 1) + 
        #             1 * abs(color_hist_avg[-1] - 1) + 
        #             20 * self.bottle_abnormal_flag
        #             )
        #     }
        # elif self.class_name == "pushpins":
        #     return {
        #         "pred_score":  torch.tensor( 
        #             5 * anomaly_map_ret_eva02.max().item() +
        #             40 * abs(color_hist_avg[-1] - 1) +
        #             20 * self.pushpins_abnormal_flag
        #             )
        #     }
        # elif self.class_name == "screw_bag":
        #     return {
        #         "pred_score":  torch.tensor( 
        #                 10 * anomaly_map_ret_dinov2.max().item() +
        #                 6 * anomaly_map_ret_eva02.max().item()   +
        #                 6 * dinov2_anomaly_map_ret_part.max().item() +
        #                 6 * dinov2_subcategory_mean_dists.max().item() +
        #                 6 * eva02_subcategory_mean_dists.max().item() +
        #                 20 * abs(area_hist_avg[-1] - 1) +
        #                 20 * abs(area_hist_avg[-2] - 1)
        #                                     )
        #     }
        # elif self.class_name == "splicing_connectors":
        #     return {
        #         "pred_score":  torch.tensor( 
        #             10 * anomaly_map_ret_dinov2.max().item() +
        #             20 * anomaly_map_ret_eva02.max().item()  +
        #             5 * dinov2_subcategory_mean_dists.max().item() +
        #             2 * eva02_subcategory_mean_dists.max().item() +
        #             30 * self.connector_abnormal_flag
        #             )
        #     }
        # else:  # "breakfast_box"
        #     return {
        #         "pred_score":  torch.tensor( 
        #             10 * anomaly_map_ret_dinov2.max().item() 
        #             + 1 * dinov2_subcategory_mean_dists.max().item() + 
        #             + 3 * abs(area_hist_avg.min() - 1) + 1 * abs(area_hist_avg.max() - 1)
        #                                     )
        #     }
    @torch.no_grad()
    def preload_heatmaps(self, stage="test", k=1):
        heatmap_cache = {}
        for cat_id in range(self.foreground_num[self.class_name]):
                train_files = [
                    f"./outputs/{self.class_name}/{stage}/{i:03d}/mask_{cat_id}.png"
                    for i in range(k)
                ]
                for path in train_files:
                    if path not in heatmap_cache:
                        img = cv2.imread(path, 0)
                        # 添加异常处理
                        if img is None:
                            raise FileNotFoundError(f"热图文件缺失: {path}")
                        heatmap_cache[path] = img
        return heatmap_cache
    
    def process_subcategory(self, cat_id, normal_dino_patches, heatmap_cache, stage="test", k=1):
        """处理单个子类别的优化版本"""
        # 使用生成器避免中间变量累积
        def get_features():
            for j in range(k):
                path = f"./outputs/{self.class_name}/{stage}/{j:03d}/mask_{cat_id}.png"
                mask = heatmap_cache[path]

                _, thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = torch.from_numpy(thresh).view(1, 1, *thresh.shape)
                mask_stack = F.interpolate( thresh, size=self.orginal_grid_size, mode="bilinear", align_corners=True ).reshape(self.orginal_grid_size**2)
                # if stage == "test" and  mask_stack.sum() < 1:
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
            F.normalize(sub_features, p=2, dim=1, out=sub_features)
        
        return sub_features, sub_patch_features
    def generate_mask(self, sample):
        masks, boxes, obj_names, pil_img = self.mask_generator.forward(sample, self.class_name)     # SAM mask
        clusted_masks, label_ids = self.mask_generator.cluster_mask(masks, boxes, obj_names, self.class_name)
        return clusted_masks, boxes, label_ids, pil_img

    def setup(self, data: dict) -> None:
        if not self.gdino_init:
            self.mask_generator = GSAM2Predictor()
            self.gdino_init = True

        few_shot_samples = data.get("few_shot_samples")
        self.class_name = data.get("dataset_category")
        # if self.class_name == "splicing_connectors":
        #     few_shot_samples = v2.functional.crop(few_shot_samples, 15, 15, 226, 226)

        self.shot = len(few_shot_samples)       # k shots
        transformed_normal_image = self.transform_dino(few_shot_samples).cuda()
        heatmap_cache = dict()
        if self.gdino_init:
            self.mask_generator.objs_dict = self.objs_dict
            self.mask_generator.foreground_num = self.foreground_num
            all_area_hists = list()
            all_color_hists = list()
            for idx, sample in enumerate(few_shot_samples):
                clusted_masks, boxes, label_ids, pil_img = self.generate_mask(sample)
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
                        
                        cv2.imwrite(path, mask*255)
            # 归一化结果
            self.class_hists_train = np.mean(np.vstack(all_area_hists), axis=0)
            self.class_color_train = np.mean(np.vstack(all_color_hists), axis=0)

        with torch.no_grad():
            self.dinov2_normal_patch_tokens = self.model_dinov2.forward_intermediates(transformed_normal_image, indices=self.out_layers, intermediates_only=True)
            self.dinov2_normal_patch_tokens = [p.permute(0, 2, 3, 1).view(self.shot, self.orginal_grid_size**2, self.embedding_dim) for p in self.dinov2_normal_patch_tokens]

            self.eva02_normal_patch_tokens = self.model_eva02.forward_intermediates(transformed_normal_image, indices=self.out_layers, intermediates_only=True)
            self.eva02_normal_patch_tokens = [p.permute(0, 2, 3, 1).view(self.shot, self.orginal_grid_size**2, self.embedding_dim) for p in self.eva02_normal_patch_tokens]
            # self.dinov2_normal_patch_tokens = (  self.dinov2_normal_patch_tokens / self.dinov2_normal_patch_tokens.norm() )
            # self.eva02_normal_patch_tokens =  (  self.eva02_normal_patch_tokens / self.eva02_normal_patch_tokens.norm() )

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
            self.part_normal_patch_tokens_eva02.append(part_normal_patch_tokens_eva02_layer)       # [4, cats, ]
            self.selected_features_eva02.append(selected_features_eva02_layer)

        self.selected_features_dinov2= []
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
            self.part_normal_patch_tokens_dinov2.append(part_normal_patch_tokens_dinov2_layer)       # [4, cats, ]
            self.selected_features_dinov2.append(selected_features_dinov2_layer)

        del heatmap_cache
