"""Model for submission."""  # add BeiT for optimization
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
import copy
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
import hashlib

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import open_clip_local as open_clip
from open_clip_local.pos_embed import get_2d_sincos_pos_embed

from accelerate import init_empty_weights

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
        self.transform_clip = v2.Compose(
            [
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )

        self.transform_BeiT = v2.Compose([
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        ])


        self.img_size = 448
        self.img_size_BeiT = 384
        # NOTE: Create your model.

        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',img_size = self.img_size) #clip model
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K') #get tokenizer of clip 

        self.feat_size = int(self.img_size // 14) 
        self.feat_size = self.feat_size * 2 
        self.ori_feat_size = int(self.img_size // 14)

        self.scale_list = [1, 3]

        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        self.checkpoint = "./checkpoint"
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        sam_filename = os.path.join(self.checkpoint, "sam_vit_h_4b8939.pth")
        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        if not os.path.exists(sam_filename):
            from anomalib.data.utils.download import DownloadProgressBar
            from urllib.request import urlretrieve
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=sam_url.split("/")[-1]) as progress_bar:
                urlretrieve(  # noqa: S310  # nosec B310
                    url=f"{sam_url}",
                    filename=sam_filename,
                    reporthook=progress_bar.update_to,
                )

        
        self.model_sam = sam_model_registry["vit_h"](checkpoint = sam_filename).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model = self.model_sam)

        self.model_clip.eval()
        self.test_args = None
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
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box', 'black background'],   # 
                        "juice_bottle": ['bottle', ['black background', 'background']],
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            #"pushpins": [['pushpin', 'pin'], ['black background', 'background']],
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
            #"pushpins": [['pushpin', 'pin'], ['black background', 'background']],
            "screw_bag": [['hex screw', 'hexagon bolt'], ['hex nut', 'hexagon nut'], ['ring washer', 'ring gasket'], ['plastic bag', 'background']],
            "splicing_connectors": [['splicing connector', 'splice connector',], ['cable', 'wire'], ['grid']],
        }
        

        self.query_threshold_dict = {
            "breakfast_box": [0., 0., 0., 0., 0., 0., 0.],
            "juice_bottle": [0., 0., 0.],
            "splicing_connectors": [0.15, 0.15, 0.15, 0., 0.],
            "pushpins": [0.2, 0., 0., 0.],
            #"pushpins": [0.1, 0., 0., 0.],
            "screw_bag": [0., 0., 0.,],
        }


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

        self.few_shot_inited = False

        from dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        # -------------------------------
        from transformers import BeitFeatureExtractor, BeitForImageClassification
        # self.model_BeiT = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384')
        # local_model_path = snapshot_download(
        #     repo_id='microsoft/beit-large-patch16-384',
        #     cache_dir=os.path.join(self.checkpoint, "cache"),
        # )
        # self.model_BeiT = BeitForImageClassification.from_pretrained(local_model_path)
        self.model_BeiT = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384', 
                                                                     cache_dir=os.path.join(self.checkpoint, "cache")
                                                                    )


        self.model_BeiT.eval().to(self.device)
        
        self.BeiT_extractor = self.BeitIntermediateFeatureExtractor(self.model_BeiT, target_layers=[5, 11, 17, 23])
        # ------------------------------

        current_path = os.path.dirname(__file__)
        pkl_file = os.path.join(current_path, "memory_bank/statistic_scores_mean.pkl")
        self.stats = pickle.load(open(pkl_file, "rb"))

        image_weights_file = os.path.join(current_path, "memory_bank/image_weights.pkl")
        self.image_weights_stats = pickle.load(open(image_weights_file, "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False #True #False

        self.cache_mode = "ram" # "disk" or "ram"
        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.sam_cache = {}

    class BeitIntermediateFeatureExtractor:
        def __init__(self, model, target_layers):
            self.model = model
            self.target_layers = target_layers
            self.outputs = []
            self.hooks = []

        def hook_fn(self, module, input, output):
            self.outputs.append(output[0].detach())

        def register_hooks(self):
            self.clear_hooks()    
            self.outputs = []      
            '''
            for i in range(4):
                block = self.model.swin.encoder.layers[i]
                handle = block.register_forward_hook(self.hook_fn)
                self.hooks.append(handle)
            '''
            for idx in self.target_layers:
                layer = self.model.beit.encoder.layer[idx]
                handle =layer.register_forward_hook(self.hook_fn)
                self.hooks.append(handle)


        def clear_hooks(self):
            for h in self.hooks:
                h.remove()
            self.hooks = []
            self.outputs = []

        def get_outputs(self):
            return self.outputs

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
        self.image_digest = None
        self.k_shot = few_shot_samples.size(0)
        self.few_shot_inited = False
        self.process(class_name, few_shot_samples)
        self.few_shot_inited = True
        with torch.no_grad():
            self.text_embedding = self.compute_text_embeddings(self.model_clip, [class_name], self.tokenizer, self.device)[class_name]


    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        return None


    def forward(self, image: torch.Tensor):
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
        # encode batch to str and md5
        self.image_digest = hashlib.md5(batch[0, 0, 128, :].cpu().numpy().tobytes()).hexdigest()
        
        self.anomaly_flag = False

        batch_resize = F.interpolate(batch, size=(self.img_size, self.img_size), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        batch_resize = self.transform_clip(batch_resize).to(self.device)

        batch_resize_BeiT = F.interpolate(batch, size=(self.img_size_BeiT, self.img_size_BeiT), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        batch_resize_BeiT =  self.transform_BeiT(batch_resize_BeiT).to(self.device)

        results = self.forward_one_sample((batch_resize, batch_resize_BeiT), self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset)

        structural_score = results['structural_score'][0]
        instance_hungarian_match_score = results['instance_hungarian_match_score']

        structural_score_raw = results['structural_score']
        text_score = results["text_score"]
        # text_score = 0
        instance_hungarian_match_score_raw = results['instance_hungarian_match_score']

        # anomaly_map_structural = results['anomaly_map_structural']

        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
        (pr_sp1_1, pr_sp1_2, pr_sp1_3) = structural_score_raw
        pr_sp2 = instance_hungarian_match_score_raw
        text_scores = text_score
        if self.class_name in self.stats and self.k_shot in self.stats[self.class_name]:
            if self.few_shot_mean in self.stats[self.class_name][self.k_shot]:
                stats = self.stats[self.class_name][self.k_shot][self.few_shot_mean]
            else:
                current_path = os.path.dirname(__file__)
                pkl_file1 = os.path.join(current_path, "memory_bank/statistic_scores.pkl")
                self.stats = pickle.load(open(pkl_file1, "rb"))
                stats = self.stats[self.class_name][self.k_shot]
            standard_pr_sp1_1 = (pr_sp1_1 - stats["pr_sp1_1"]["mean"]) / stats["pr_sp1_1"]["unbiased_std"]
            standard_pr_sp1_2 = (pr_sp1_2 - stats["pr_sp1_2"]["mean"]) / stats["pr_sp1_2"]["unbiased_std"]
            standard_pr_sp1_3 = (pr_sp1_3 - stats["pr_sp1_3"]["mean"]) / stats["pr_sp1_3"]["unbiased_std"]
            standard_text_scores = (text_scores - stats["text_scores"]["mean"]) / stats["text_scores"]["unbiased_std"]
            if self.class_name == 'pushpins':
                standard_pr_sp2 = 0
            else:
                standard_pr_sp2 = (pr_sp2 - stats["pr_sp2"]["mean"]) / stats["pr_sp2"]["unbiased_std"]
        
        if self.weights is not None:
            alpha_stu_weights = self.weights["alpha_stu_weights"]
            alpha_sp = self.weights["alpha_sp"]
            alpha_text = self.weights["alpha_text"]
            
            structural_score = alpha_stu_weights[0] * standard_pr_sp1_1 + alpha_stu_weights[1] * standard_pr_sp1_2 + alpha_stu_weights[2] * standard_pr_sp1_3

            pr_sp = np.max(np.stack([structural_score, alpha_sp * standard_pr_sp2], axis=0), axis=0)
            pr_sp = sigmoid(pr_sp + alpha_text * standard_text_scores)
        else:
            pr_sp = np.max(np.stack([standard_pr_sp1_2  , standard_pr_sp2], axis=0), axis=0)
            pr_sp = sigmoid(pr_sp)
        
        pred_score = pr_sp

        if self.anomaly_flag:
            pred_score = 1.
            self.anomaly_flag = False
        
        return ImageBatch(
            image=image,
            pred_score=torch.tensor(pred_score).to(image.device),
        )

    
    from prompt_ensemble import encode_text_with_prompt_ensemble
    def compute_text_embeddings(self, model, objs, tokenizer, device):
        output_text = encode_text_with_prompt_ensemble(model, objs, tokenizer, device)
        return output_text
    
    def Padding_same(self, x, kernel_size):
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        pad_top = (kernel_size - 1) // 2
        pad_bottom = kernel_size // 2
        padded_input = F.pad(x, pad=(pad_left, pad_right, pad_top, pad_bottom), mode= "constant")  # mode='replicate'
        padded_input = F.avg_pool2d(padded_input, kernel_size=kernel_size, stride=1, padding=0)
        return padded_input
    
    def Process_support(self, good_samples, scale_list): 
        good_list = []
        index_num_list_1 = []
        B, L, C = good_samples[0].shape
        H = int(np.sqrt(L)) 
        for good in good_samples:
            good_scale_list = []
            good = good.permute(0, 2, 1).view(B,-1,H,H)
            for scale in scale_list:
                if scale != 1:
                    good_temp =  self.Padding_same(good.clone(),  kernel_size= scale)
                else:
                    good_temp = good.clone() 
                good_temp = good_temp.view(B, C, -1).permute(0, 2, 1).reshape(-1, C).unsqueeze(0)
                U, S, V = torch.svd(good_temp.permute(0,2,1))
                total_energy = torch.sum(S)
                energy_threshold = 0.6 * total_energy
                cumulative_energy = torch.cumsum(S[0,:], dim=0)
                selected_index = torch.where(cumulative_energy >= energy_threshold)[0][0]
                index_num_list_1.append(selected_index.cpu().numpy())

                bias = torch.ones((U.size(0), U.size(1),1), device=U.device, dtype= torch.float)
                good_temp = U[:,:,:selected_index].permute(0,2,1)
                good_scale_list.append(good_temp)
            good_list.append(good_scale_list)
        return good_list
    
    def norm_patch(self, patches, is_forget = False):
        if is_forget:
            patches = patches[:,1:,:]
        mean = torch.mean(patches, dim=1, keepdim=True)  
        std = torch.std(patches, dim=1, keepdim=True)   
        patches = (patches - mean) / std
        return patches

    def set_viz(self, viz):
        self.visualization = viz


    def set_val(self, val):
        self.validation = val


    def forward_one_sample(self, batchs, mem_patch_feature_clip_coreset: torch.Tensor, mem_patch_feature_dinov2_coreset: torch.Tensor):
        batch, batch_BeiT = batchs[0], batchs[1]
        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(batch, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_clip = patch_tokens_clip.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)

            patch_tokens_test = copy.deepcopy(patch_tokens_clip)
            patch_tokens_test = patch_tokens_test.permute(0, 2, 3, 1).view(patch_tokens_test.shape[0], -1, self.vision_width * len(self.feature_list))
            patch_tokens_test = list(patch_tokens_test.chunk(len(self.feature_list), dim=-1))

            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1) 

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ self.text_embedding).softmax(dim=-1)
            pr = text_probs[0][1].cpu().item()
            

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(batch, out_layer_list=self.feature_list_dinov2)
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)

            patch_tokens_dinov2_test = copy.deepcopy(patch_tokens_dinov2)
            patch_tokens_dinov2_test = patch_tokens_dinov2_test.permute(0, 2, 3, 1).view(patch_tokens_dinov2_test.shape[0], -1, self.vision_width * len(self.feature_list))
            patch_tokens_dinov2_test = list(patch_tokens_dinov2_test.chunk(len(self.feature_list), dim=-1))


            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1) # (1x64x64, 1024x4)

        with torch.no_grad():
            self.BeiT_extractor.register_hooks()
            _ = self.model_BeiT(pixel_values = batch_BeiT)
            patch_tokens_BeiT = self.BeiT_extractor.get_outputs()
            self.BeiT_extractor.clear_hooks()
            patch_tokens_BeiT = [p[:, 1:, :] for p in patch_tokens_BeiT] 
            patch_tokens_BeiT = torch.cat(patch_tokens_BeiT, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_BeiT = patch_tokens_BeiT.view(1, self.img_size_BeiT // 16, self.img_size_BeiT // 16, -1).permute(0, 3, 1, 2)
            patch_tokens_BeiT = F.interpolate(patch_tokens_BeiT, size=(self.img_size_BeiT // 16, self.img_size_BeiT // 16), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_BeiT_test = copy.deepcopy(patch_tokens_BeiT)
            patch_tokens_BeiT_test = patch_tokens_BeiT_test.permute(0, 2, 3, 1).view(patch_tokens_BeiT_test.shape[0], -1, self.vision_width * len(self.feature_list))
            patch_tokens_BeiT_test = list(patch_tokens_BeiT_test.chunk(len(self.feature_list), dim=-1))

        
        '''adding for kmeans seg '''
        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(self.feat_size * self.feat_size, self.embed_dim)
        else:
            proj_patch_tokens = proj_patch_tokens.view(self.feat_size * self.feat_size, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        mid_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            mid_features = temp_feat if mid_features is None else torch.cat((mid_features, temp_feat), -1)
            
        if self.feat_size != self.ori_feat_size:
            mid_features = mid_features.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            mid_features = F.interpolate(mid_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            mid_features = mid_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        else:
            mid_features = mid_features.view(-1, self.vision_width * len(self.cluster_feature_id))
        mid_features = F.normalize(mid_features, p=2, dim=-1)
             
        # self.test_idx += 1
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name, test_mode=True)

        '''calculate patchcore'''
        anomaly_maps_patchcore_1 = []
        anomaly_maps_patchcore_2 = []
        anomaly_maps_patchcore_3 = []

        if self.class_name in ['splicing_connectors', 'pushpins', 'screw_bag','juice_bottle', 'breakfast_box']: # clip feature for patchcore  
            len_feature_list = len(self.feature_list)
            for i in range(len(patch_tokens_test)):
                patch_single_layer = patch_tokens_test[i]
                patch_single_layer = self.norm_patch(patch_single_layer)
                B, L, C = patch_single_layer.shape
                H = int(np.sqrt(L))
                patch_single_layer = patch_single_layer.permute(0, 2, 1).view(B,-1,H,H)
                for j in range(len(self.scale_list)):
                    if self.scale_list[j] != 1:
                        patch_single_layer_temp =  self.Padding_same(patch_single_layer.clone(),  kernel_size= self.scale_list[j])
                    else:
                        patch_single_layer_temp = patch_single_layer.clone()
                    patch_single_layer_temp = patch_single_layer_temp.view(B,C,-1).permute(0, 2, 1)
                    W = torch.bmm(patch_single_layer_temp, self.patch_normal_CLIP[i][j].permute(0,2,1))
                    Q = torch.bmm(W, self.patch_normal_CLIP[i][j])
                    sim_result =  F.cosine_similarity(patch_single_layer_temp , Q, dim=-1).squeeze().cpu().numpy()
                    anomaly_maps_patchcore_1.append(1 - (sim_result+1)*0.5)
        
        
        if self.class_name in ['splicing_connectors', 'pushpins', 'screw_bag','juice_bottle', 'breakfast_box']: # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for i in range(len(patch_tokens_dinov2_test)):
                patch_single_layer = patch_tokens_dinov2_test[i]
                patch_single_layer = self.norm_patch(patch_single_layer)
                B, L, C = patch_single_layer.shape
                H = int(np.sqrt(L))
                patch_single_layer = patch_single_layer.permute(0, 2, 1).view(B,-1,H,H)
                for j in range(len(self.scale_list)):
                    if self.scale_list[j] != 1:
                        patch_single_layer_temp =  self.Padding_same(patch_single_layer.clone(),  kernel_size= self.scale_list[j])
                    else:
                        patch_single_layer_temp = patch_single_layer.clone()
                    patch_single_layer_temp = patch_single_layer_temp.view(B,C,-1).permute(0, 2, 1)
                    W = torch.bmm(patch_single_layer_temp, self.patch_normal_dinov2[i][j].permute(0,2,1))
                    Q = torch.bmm(W, self.patch_normal_dinov2[i][j])
                    sim_result =  F.cosine_similarity(patch_single_layer_temp , Q, dim=-1).squeeze().cpu().numpy()
                    anomaly_maps_patchcore_2.append(1 - (sim_result+1)*0.5)

        
        if self.class_name in ['splicing_connectors', 'pushpins', 'screw_bag','juice_bottle', 'breakfast_box']: # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for i in range(len(patch_tokens_BeiT_test)):
                patch_single_layer = patch_tokens_BeiT_test[i]
                patch_single_layer = self.norm_patch(patch_single_layer)
                B, L, C = patch_single_layer.shape
                H = int(np.sqrt(L))
                patch_single_layer = patch_single_layer.permute(0, 2, 1).view(B,-1,H,H)
                for j in range(len(self.scale_list)):
                    if self.scale_list[j] != 1:
                        patch_single_layer_temp =  self.Padding_same(patch_single_layer.clone(),  kernel_size= self.scale_list[j])
                    else:
                        patch_single_layer_temp = patch_single_layer.clone()
                    patch_single_layer_temp = patch_single_layer_temp.view(B,C,-1).permute(0, 2, 1)
                    W = torch.bmm(patch_single_layer_temp, self.patch_normal_BeiT[i][j].permute(0,2,1))
                    Q = torch.bmm(W, self.patch_normal_BeiT[i][j])
                    sim_result =  F.cosine_similarity(patch_single_layer_temp , Q, dim=-1).squeeze().cpu().numpy()
                    anomaly_maps_patchcore_3.append(1 - (sim_result+1)*0.5)
        

        structural_score_1 = np.stack(anomaly_maps_patchcore_1).mean(0).max()     #  
        structural_score_2 = np.stack(anomaly_maps_patchcore_2).mean(0).max() 
        structural_score_3 = np.stack(anomaly_maps_patchcore_3).mean(0).max() 

        if self.class_name != 'pushpins':
            instance_masks = results["instance_masks"]   
            anomaly_instances_hungarian = []
            instance_hungarian_match_score = 1.
            if self.mem_instance_masks is not None and len(instance_masks) != 0:  
                for patch_feature, batch_mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                    instance_features = [patch_feature[mask, :].mean(0, keepdim=True) for mask in instance_masks]
                    if not instance_features:
                        continue
                    instance_features = torch.cat(instance_features, dim=0)
                    instance_features = F.normalize(instance_features, dim=-1)
                    mem_instance_features = []
                    for mem_patch_feature, mem_instance_masks in zip(batch_mem_patch_feature.chunk(self.k_shot), self.mem_instance_masks):
                        mem_instance_features.extend([mem_patch_feature[mask, :].mean(0, keepdim=True) for mask in mem_instance_masks]) 
                    if not mem_instance_features:
                        continue
                    mem_instance_features = torch.cat(mem_instance_features, dim=0)
                    mem_instance_features = F.normalize(mem_instance_features, dim=-1)

                    normal_instance_hungarian = (instance_features @ mem_instance_features.T)
                    cost_matrix = (1 - normal_instance_hungarian).cpu().numpy()
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix) 
                    cost = cost_matrix[row_ind, col_ind].sum() 
                    cost = cost / min(cost_matrix.shape)
                    anomaly_instances_hungarian.append(cost)

                instance_hungarian_match_score = np.mean(anomaly_instances_hungarian)  
        else:
            instance_hungarian_match_score = 0  
           
        results = {"text_score": pr, 'structural_score': (structural_score_1, structural_score_2, structural_score_3),  'instance_hungarian_match_score': instance_hungarian_match_score}

        return results


    def histogram(self, image, cluster_feature, proj_patch_token, class_name, test_mode=None):
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

        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        self.anomaly_flag = False
        instance_masks = []
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

            img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            _ , s_channel, v_channel = cv2.split(hsv_img)

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
                
                selected_area_s= []
                for lab_s in range(1, num_lab_s):
                   area_s = stats_s[lab_s, cv2.CC_STAT_AREA]  
                   ratio2 = area_s / self.median_area_s_save
                   if 0.7 <= ratio2 <= 1.3: 
                      selected_area_s.append(area_s)
                if len(selected_area_s) != self.pushpins_count: 
                   self.anomaly_flag = True
            
            gray_no_background = img_gray[img_gray != 0]
            pushpins_min_range_pixel_value = np.percentile(gray_no_background, 15)
            thresh = cv2.inRange(img_gray, pushpins_min_range_pixel_value, 255)
            each_foreground_pins_counts = split_and_check_foreground(thresh, thresh_pins, rows=3, cols=5) 
            
            if self.few_shot_inited and self.each_foreground_pins_counts != 0 and self.anomaly_flag is False:
                for i , each_counts in enumerate(each_foreground_pins_counts):
                    ratio = each_counts / self.each_foreground_pins_counts
                    if (ratio > 1.6 or ratio < 0.4): 
                        self.anomaly_flag = True
                        break
        
            if not self.few_shot_inited:
                return  {"avg_area_r": avg_area_r, "median_area_s": median_area_s, "each_foreground_pins_counts": each_foreground_pins_counts}
        
        elif self.class_name == 'splicing_connectors':
            sam_mask, sam_mask_max_area = self.get_sam_masks(raw_image, self.class_name, test_mode=test_mode)

            patch_similarity = (proj_patch_token @ self.patch_query_obj.T)  
            patch_mask = patch_similarity.argmax(-1) 
            patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)
            

            # filter small region for patch merge sam
            binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 32: 
                    patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background

            def detect_slot_anomalies_distance(binary_clamps_image, binary_cable_image, image_width):
                contours, _ = cv2.findContours(binary_clamps_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                left_corners = []
                right_corners = []

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 250:
                        continue
                    
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect).astype(int)
                    cx = np.mean(box[:, 0])
                    
                    if cx < image_width / 2:
                        top_right = max(box, key=lambda p: p[0] - p[1])
                        left_corners.append((tuple(top_right), area))
                    else:
                        top_left = min(box, key=lambda p: p[0] + p[1])
                        right_corners.append((tuple(top_left), area))

                if not left_corners or not right_corners:
                    return -1, -1, -1

                left_corners.sort(key=lambda x: x[1], reverse=True)
                right_corners.sort(key=lambda x: x[1], reverse=True)
                left_corner = left_corners[0][0]
                right_corner = right_corners[0][0]

                kernel = np.ones((3, 3), np.uint8)
                opened = cv2.morphologyEx(binary_cable_image, cv2.MORPH_OPEN, kernel, iterations=2)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return -1, -1, -1

                cable_contour = max(contours, key=cv2.contourArea)
                cable_points = cable_contour[:, 0, :]

                left_end_idx = np.argmin(cable_points[:, 0])
                right_end_idx = np.argmax(cable_points[:, 0])
                left_end = tuple(cable_points[left_end_idx])
                right_end = tuple(cable_points[right_end_idx])

                if abs(left_end[0] - left_corner[0]) > image_width / 4 or abs(right_end[0] - right_corner[0]) > image_width / 4:
                    return -1, -1, -1

                left_y_distance = abs(left_end[1] - left_corner[1])
                right_y_distance = abs(right_end[1] - right_corner[1])

                if right_y_distance == 0:
                    return -1, left_y_distance, right_y_distance

                distance_ratio = left_y_distance / right_y_distance

                return distance_ratio, left_y_distance, right_y_distance
        
            #  object count hist for default
            binary = (sam_mask_max_area == 0).astype(np.uint8) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: 
                    binary[temp_mask] = 0 
                else:
                    count += 1
            if count != 1 and self.anomaly_flag is False: 
                self.anomaly_flag = True

            patch_merge_sam[~(binary.astype(np.bool_))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            # erode the cable and divide into left and right parts
            kernel = np.ones((23, 23), dtype=np.uint8)
            erode_binary = cv2.erode(binary, kernel)
            h, w = erode_binary.shape
            distance = 0

            left, right = erode_binary[:, :int(w/2)],  erode_binary[:, int(w/2):]   
            left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
            right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1] # foreground

            binary_cable = (patch_merge_sam == 1).astype(np.uint8)
            binary_cable_orinal = binary_cable.copy()
            
            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_cable = cv2.erode(binary_cable, kernel)
            kernel1 = np.ones((5, 5),  dtype=np.uint8)
            binary_cable1 = cv2.erode(binary_cable, kernel1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cable, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:
                    binary_cable[temp_mask] = 0 # set to background

            binary_cable = cv2.resize(binary_cable, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            
            binary_clamps = (patch_merge_sam == 0).astype(np.uint8)
            binary_clamps_orinal = binary_clamps.copy()

            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_clamps = cv2.erode(binary_clamps, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clamps, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64:
                    binary_clamps[temp_mask] = 0 # set to background
                else:
                    instance_mask = temp_mask.astype(np.uint8)
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                    if instance_mask.any():
                        instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

            binary_clamps = cv2.resize(binary_clamps, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

            binary_connector = cv2.resize(binary, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            
            query_cable_color = encode_obj_text(self.model_clip, self.splicing_connectors_cable_color_query_words_dict, self.tokenizer, self.device)
            cable_feature = proj_patch_token[binary_cable.astype(np.bool_).reshape(-1), :].mean(0, keepdim=True)
            idx_color = (cable_feature @ query_cable_color.T).argmax(-1).squeeze(0).item()
            foreground_pixel_count = np.sum(erode_binary) / self.splicing_connectors_count[idx_color]

            #the middle of cabel
            slice_cable = binary_cable1[:, int(w/2)-5: int(w/2)+5]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
            cable_count = num_labels - 1
            if cable_count != 1 and self.anomaly_flag is False: # two cables
                self.anomaly_flag = True

            # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:    # color and number mismatch
                    self.anomaly_flag = True

            # left right hist for symmetry
            ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
            if self.few_shot_inited and (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False: # left right asymmetry in clamp
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
                    if ratio < 0.6 or ratio > 1.4:  
                        self.anomaly_flag = True


            if not self.few_shot_inited:
                distance_ratio, _, _= detect_slot_anomalies_distance(binary_clamps_orinal, binary_cable_orinal, w)

            if self.few_shot_inited and self.distance_ratio_save != 0 and  self.anomaly_flag is False:
                distance_ratio, _, _ = detect_slot_anomalies_distance(binary_clamps_orinal, binary_cable_orinal, w)
                if distance_ratio != -1:
                    if 0.1 < distance_ratio < 0.4 or distance_ratio > 1.28:
                        self.anomaly_flag = True
                                 
            # todo    mismatch cable link  
            binary_foreground = binary.astype(np.uint8)
            if binary_connector.any():
                instance_masks.append(binary_connector.astype(np.bool_).reshape(-1))
            if binary_clamps.any():
                instance_masks.append(binary_clamps.astype(np.bool_).reshape(-1))
            if binary_cable.any():
                instance_masks.append(binary_cable.astype(np.bool_).reshape(-1))      

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if not self.few_shot_inited:
                return {"distance": distance, "instance_masks": instance_masks, "foreground_pixel_count": foreground_pixel_count,  "distance_ratio":distance_ratio}
            else:
                return {"distance": distance, "instance_masks": instance_masks}
           
        elif self.class_name == 'screw_bag':
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

            sam_mask, sam_mask_max_area = self.get_sam_masks(raw_image, self.class_name, test_mode=test_mode)

            kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            patch_similarity = (proj_patch_token @ self.patch_query_obj.T) 
            patch_mask = patch_similarity.argmax(-1)  
            patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)
            
            # filter small region for patch merge sam
            binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 32: # 448x448
                    patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background
            
            # pixel hist of kmeans mask
            foreground_pixel_count = np.sum(np.bincount(kmeans_mask.reshape(-1))[:len(self.foreground_label_idx[self.class_name])])  # foreground pixel
            if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist
                # todo: optimize
                if ratio < 0.96 or ratio > 1.06:  #if ratio < 0.94 or ratio > 1.06
                    self.anomaly_flag = True
            
            img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            color =cv2.resize(img_bgr, self.screw_bag_cicrle_image_shape, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            if self.few_shot_inited and self.anomaly_flag is False:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=150, param2=18, minRadius=self.screw_bag_circle_radius[0], maxRadius=self.screw_bag_circle_radius[1])
                if circles is not None :
                    circles = np.uint16(np.around(circles))
                    if(circles.shape[1] != self.screw_bag_circle_count):
                        self.anomaly_flag = True
            
            thresh = cv2.inRange(gray, 19, 255)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            selected_mask = np.zeros_like(thresh, dtype=np.uint8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area_index = np.argmax(areas) + 1
            selected_mask[labels == max_area_index] = 255 
            contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:    
                Area_s = cv2.countNonZero(selected_mask) 
                filled_mask = selected_mask.copy()
                contours, hierarchy = cv2.findContours(filled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for i, cnt in enumerate(contours):
                    if hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1:
                        cv2.drawContours(filled_mask, [cnt], 0, 255, -1)   
                Area_f = cv2.countNonZero(filled_mask) 
                if self.few_shot_inited and self.anomaly_flag is False:
                    if (Area_f - Area_s) > 150:
                        self.anomaly_flag = True
                    
                contours_filled = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                max_contour_filled = max(contours_filled, key=cv2.contourArea)
                Area_c = cv2.contourArea(max_contour_filled)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                closed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
                contours_closed = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                max_contour_closed = max(contours_closed, key=cv2.contourArea)
                Area_ss = cv2.contourArea(max_contour_closed)
                if not self.few_shot_inited:
                    area_difference_ss_c = Area_ss - Area_c
                if self.few_shot_inited and self.area_difference_ss_c_save != 0 and self.anomaly_flag is False:
                    ratio2 = (Area_ss - Area_c) / self.area_difference_ss_c_save
                    if ratio2 > 1.5:
                        self.anomaly_flag = True
            
            # patch hist
            binary_screw = np.isin(kmeans_mask, self.foreground_label_idx[self.class_name])

            resized_binary_screw = cv2.resize(binary_screw.astype(np.uint8), (patch_merge_sam.shape[1], patch_merge_sam.shape[0]), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam[~(resized_binary_screw.astype(np.bool_))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            # # todo: count of screw, nut and washer, screw of different length
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]
            
            if not self.few_shot_inited:
                return {"foreground_pixel_count": foreground_pixel_count,"instance_masks": instance_masks, "area_difference_ss_c": area_difference_ss_c}
            else:
                return {"foreground_pixel_count": foreground_pixel_count,"instance_masks": instance_masks}

        elif self.class_name == 'breakfast_box': 
            sam_mask, sam_mask_max_area = self.get_sam_masks(raw_image, self.class_name, test_mode=test_mode)

            patch_similarity = (proj_patch_token @ self.patch_query_obj.T)  
            patch_mask = patch_similarity.argmax(-1)  
            patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)

            # filter small region for patch merge sam
            binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 32: # 448x448
                    patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background

            # todo: exist of foreground
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8) 

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            return {"instance_masks": instance_masks}
        
        elif self.class_name == 'juice_bottle': 
            sam_mask, sam_mask_max_area = self.get_sam_masks(raw_image, self.class_name, test_mode=test_mode)

            patch_similarity = (proj_patch_token @ self.patch_query_obj.T) 
            patch_mask = patch_similarity.argmax(-1) 
            patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

            resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)

            # filter small region for patch merge sam
            binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 32: # 448x448
                    patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background
            
            patch_merge_sam[sam_mask == 0] = self.patch_query_obj.shape[0] - 1  # 79.5

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
                self.anomaly_flag = True
            
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1) ).astype(np.uint8) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks)
            return {"instance_masks": instance_masks}

        return {"instance_masks": instance_masks}


    def process_k_shot(self, class_name, few_shot_samples):
        # few_shot_samples = F.interpolate(few_shot_samples, size=(self.img_size, self.img_size), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        few_shot_sample, few_shot_sample_BeiT = few_shot_samples[0], few_shot_samples[1]
        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(few_shot_sample, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]  
            #patch_tokens_temp = copy.deepcopy(patch_tokens)
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (bs, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (bs, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)

            patch_tokens_temp = copy.deepcopy(patch_tokens_clip)
            patch_tokens_temp = patch_tokens_temp.permute(0, 2, 3, 1).view(patch_tokens_temp.shape[0], -1, self.vision_width * len(self.feature_list))
            patch_tokens_temp = list(patch_tokens_temp.chunk(len(self.feature_list), dim=-1))
            patch_tokens_temp = [self.norm_patch(patch) for patch in patch_tokens_temp]


            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1)  # (bsx64x64, 1024x4)

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(few_shot_sample, out_layer_list=self.feature_list_dinov2)  # 4 x [bs, 32x32, 1024]
            # patch_tokens_dinov2_temp  = copy.deepcopy(patch_tokens_dinov2)
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (bs, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)


            patch_tokens_dinov2_temp = copy.deepcopy(patch_tokens_dinov2)
            patch_tokens_dinov2_temp = patch_tokens_dinov2_temp.permute(0, 2, 3, 1).view(patch_tokens_dinov2_temp.shape[0], -1, self.vision_width * len(self.feature_list_dinov2))
            patch_tokens_dinov2_temp = list(patch_tokens_dinov2_temp.chunk(len(self.feature_list_dinov2), dim=-1))
            patch_tokens_dinov2_temp = [self.norm_patch(patch) for patch in patch_tokens_dinov2_temp]


            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (bsx64x64, 1024x4)


        with torch.no_grad():
            self.BeiT_extractor.register_hooks()
            _ = self.model_BeiT(pixel_values = few_shot_sample_BeiT)
            patch_tokens_BeiT = self.BeiT_extractor.get_outputs()
            self.BeiT_extractor.clear_hooks()
            patch_tokens_BeiT = [p[:, 1:, :] for p in patch_tokens_BeiT] 
            patch_tokens_BeiT = torch.cat(patch_tokens_BeiT, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_BeiT = patch_tokens_BeiT.view(self.k_shot, self.img_size_BeiT // 16, self.img_size_BeiT // 16, -1).permute(0, 3, 1, 2)
            patch_tokens_BeiT = F.interpolate(patch_tokens_BeiT, size=(self.img_size_BeiT // 16,self.img_size_BeiT// 16), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_BeiT_test = copy.deepcopy(patch_tokens_BeiT)
            patch_tokens_BeiT_test = patch_tokens_BeiT_test.permute(0, 2, 3, 1).view(patch_tokens_BeiT_test.shape[0], -1, self.vision_width * len(self.feature_list))
            patch_tokens_BeiT_test = list(patch_tokens_BeiT_test.chunk(len(self.feature_list), dim=-1))
            patch_tokens_BeiT_test = [self.norm_patch(patch) for patch in patch_tokens_BeiT_test]

        
        self.patch_normal_CLIP = self.Process_support(patch_tokens_temp, scale_list= self.scale_list)
        self.patch_normal_dinov2 = self.Process_support(patch_tokens_dinov2_temp, scale_list= self.scale_list)
        self.patch_normal_BeiT = self.Process_support(patch_tokens_BeiT_test, scale_list= self.scale_list)

        cluster_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            cluster_features = temp_feat if cluster_features is None else torch.cat((cluster_features, temp_feat), 1)
        if self.feat_size != self.ori_feat_size:
            cluster_features = cluster_features.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            cluster_features = F.interpolate(cluster_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            cluster_features = cluster_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        else:
            cluster_features = cluster_features.view(-1, self.vision_width * len(self.cluster_feature_id))
        cluster_features = F.normalize(cluster_features, p=2, dim=-1)

        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim)
        else:
            proj_patch_tokens = proj_patch_tokens.view(-1, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        num_clusters = self.cluster_num_dict[class_name]
        _, self.cluster_centers = kmeans(X=cluster_features, num_clusters=num_clusters, device=self.device)
    
        self.query_obj = encode_obj_text(self.model_clip, self.query_words_dict[class_name], self.tokenizer, self.device)
        self.patch_query_obj = encode_obj_text(self.model_clip, self.patch_query_words_dict[class_name], self.tokenizer, self.device)
        self.classes = self.query_obj.shape[0]

        foreground_pixel_hist = []
        splicing_connectors_distance = []
        mem_instance_masks = []
        avg_area_r_save = []
        median_area_s_save = []
        distance_ratio_save = []
        each_foreground_pins_counts = []
        area_difference_ss_c_save = []
            
        for image, cluster_feature, proj_patch_token in zip(few_shot_sample.chunk(self.k_shot), cluster_features.chunk(self.k_shot), proj_patch_tokens.chunk(self.k_shot)):        
            self.anomaly_flag = False
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name)
            if self.class_name == 'pushpins':
                avg_area_r_save.append(results['avg_area_r'])
                median_area_s_save.append(results['median_area_s'])
                each_foreground_pins_counts.append(results['each_foreground_pins_counts'])

            elif self.class_name == 'splicing_connectors':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                splicing_connectors_distance.append(results["distance"])
                mem_instance_masks.append(results['instance_masks'])
                distance_ratio_save.append(results['distance_ratio'] if results['distance_ratio'] != -1 else 1)

            elif self.class_name == 'screw_bag':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                mem_instance_masks.append(results['instance_masks'])
                area_difference_ss_c_save.append(results["area_difference_ss_c"])

            elif self.class_name == 'breakfast_box':
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'juice_bottle':
                mem_instance_masks.append(results['instance_masks'])

        
        # pushpins
        if len(avg_area_r_save) != 0:
            self.avg_area_r_save = np.mean(avg_area_r_save)
        if len(median_area_s_save) != 0:
            self.median_area_s_save = np.mean(median_area_s_save)
        if len(each_foreground_pins_counts) != 0:
            flattened_foreground_pixel_hist = np.concatenate(each_foreground_pins_counts)
            self.each_foreground_pins_counts = np.mean(flattened_foreground_pixel_hist)

        # splicing_connectors
        if len(distance_ratio_save) != 0:
            self.distance_ratio_save = np.mean(distance_ratio_save) 

        # screw_bag
        if len(area_difference_ss_c_save) != 0:
            self.area_difference_ss_c_save = np.mean(area_difference_ss_c_save) 

        if len(foreground_pixel_hist) != 0:
            self.foreground_pixel_hist = np.mean(foreground_pixel_hist)
        if len(splicing_connectors_distance) != 0:
            self.splicing_connectors_distance = np.mean(splicing_connectors_distance)

        if len(mem_instance_masks) != 0:
            self.mem_instance_masks = mem_instance_masks

        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        return mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset

    
    # def process(self, class_name: str, few_shot_samples: list[torch.Tensor], few_shot_paths: list[str]):
    def process(self, class_name: str, few_shot_samples: list[torch.Tensor]):
        few_shot_samples_resize = F.interpolate(few_shot_samples, size=(self.img_size, self.img_size), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        few_shot_samples_resize = self.transform_clip(few_shot_samples_resize).to(self.device)

        few_shot_samples_resize_BeiT = F.interpolate(few_shot_samples, size=(self.img_size_BeiT, self.img_size_BeiT), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)
        few_shot_samples_resize_BeiT = self.transform_BeiT(few_shot_samples_resize_BeiT).to(self.device)

        self.few_shot_mean = few_shot_samples.mean().item()
        
        if class_name in self.image_weights_stats and self.k_shot in self.image_weights_stats[class_name]:
            available_means = list(self.image_weights_stats[class_name][self.k_shot].keys())
            closest_mean = min(available_means, key=lambda x: abs(x - self.few_shot_mean))
            self.weights = self.image_weights_stats[class_name][self.k_shot][closest_mean]
        else:
            self.weights = None
        
        self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset = self.process_k_shot(class_name, (few_shot_samples_resize,few_shot_samples_resize_BeiT))


    def get_sam_masks(self, raw_image: np.ndarray, class_name: str, test_mode: int = None) -> list:
        def plot_results_only(sorted_anns):
            cur = 1
            img_color = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            for ann in sorted_anns:
                m = ann['segmentation']
                img_color[m] = cur
                cur += 1
            return img_color
        
        def gen_sam_masks(raw_image):
            masks = self.mask_generator.generate(raw_image)
            sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            sam_mask = plot_results_only(sorted_masks).astype(int)
            sam_mask_max_area = sorted_masks[0]['segmentation']
            return sam_mask, sam_mask_max_area

        if test_mode is None:
            # for few shot
            sam_mask, sam_mask_max_area = gen_sam_masks(raw_image)
            
        else:
            mask_key = f"{class_name}_{self.image_digest}"

            if self.cache_mode == "ram":
                if mask_key in self.sam_cache.keys():
                    sam_mask = self.sam_cache[mask_key]['sam_mask']
                    sam_mask_max_area = self.sam_cache[mask_key]['sam_mask_max_area']
                else:
                    sam_mask, sam_mask_max_area = gen_sam_masks(raw_image)
                    self.sam_cache[mask_key] = {'sam_mask': sam_mask, 'sam_mask_max_area': sam_mask_max_area}

            elif self.cache_mode == "disk":
                cache_file = os.path.join(self.cache_dir, f"{mask_key}.pkl")
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        data_dict = pickle.load(f)
                        sam_mask = data_dict['sam_mask']
                        sam_mask_max_area = data_dict['sam_mask_max_area']
                else:
                    sam_mask, sam_mask_max_area = gen_sam_masks(raw_image)
                    with open(cache_file, 'wb') as f:
                        pickle.dump({'sam_mask': sam_mask, 'sam_mask_max_area': sam_mask_max_area}, f)
        
        return sam_mask, sam_mask_max_area
