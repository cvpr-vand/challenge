"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn

import warnings
warnings.filterwarnings("ignore")

import json
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess

from .utils.sampler import GreedyCoresetSampler
from .models import clip as open_clip
import os
from torchvision.transforms.v2.functional import resize, crop, rotate, InterpolationMode

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    
from .models.component_segmentaion import (
    split_masks_from_one_mask,
    split_masks_from_one_mask_sort,
    filter_by_combine,
    split_masks_by_connected_component,
    turn_binary_to_int,
    merge_masks,
    filter_masks_below_area,
    post_process_masks,
    compute_logical_score
)

from PIL import Image


from .models.segment_anything import (
    sam_hq_model_registry,
    SamPredictor,
)

from urllib.request import urlretrieve

class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayer, self).__init__()

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i][:, 1:, :]
            else:
                assert 0 == 1
        return tokens



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]

def save_tensor_as_jpg(tensor_images, save_dir="saved_images", class_name=None):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历每张图像并保存
    for i in range(tensor_images.shape[0]):
        # 方法1：使用 torchvision 的 save_image
        data = tensor_images[i]
        data = data.cpu().numpy()
        data = np.clip(data * 255, 0, 255).astype(np.uint8)
        data = data.transpose(1, 2, 0)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        if class_name == "pushpins":
            black_image = np.zeros_like(data)

            x1, y1, x2, y2 = 10, 15, 246, 246
            if data.shape[0] > 400:
                x1, y1, x2, y2 = 20, 30, 428, 428

            # 确保坐标不超出图像边界
            x2 = min(x2, data.shape[1])
            y2 = min(y2, data.shape[0])

            # 复制指定区域
            black_image[y1:y2, x1:x2] = data[y1:y2, x1:x2]
            data = black_image

        cv2.imwrite(f"{save_dir}/image_{i}.jpg", data)

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    return image_pil, None

def infer_dino(img, queries, box_threshold, text_threshold, area_threshold, dino_model, dino_processor, device):
    
    width, height = img.size[:2]
    img_area = width * height

    target_sizes = [(width, height)]
    inputs = dino_processor(text=queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)
        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        results = dino_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    
    # 过滤掉面积过大的检测框
    filtered_results = []
    for result in results:
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []
        
        for i, box in enumerate(result["boxes"]):
            # 计算框的面积
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            
            # 如果框的面积小于整张图像的80%，保留
            if box_area / img_area < area_threshold:
                filtered_boxes.append(box)
                filtered_labels.append(result["labels"][i])
                filtered_scores.append(result["scores"][i])
        
        filtered_result = {
            "boxes": torch.stack(filtered_boxes) if filtered_boxes else torch.zeros((0, 4)),
            "labels": filtered_labels,
            "scores": filtered_scores,

        }
        filtered_results.append(filtered_result)
    
    return filtered_results



class Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not os.path.exists("./src/eval/submission/ckpts/sam_hq_vit_h.pth"):
            urlretrieve(  # noqa: S310  # nosec B310
                    url="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
                    filename="./src/eval/submission/ckpts/sam_hq_vit_h.pth",
                )

        clip_name = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
        self.image_size = 448
        device = torch.device("cuda")
        self.out_layers = [6, 12, 18, 24]

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, self.image_size
        )
        self.clip_model.to(device)
        self.clip_model.eval()

        self.dinov2_net = torch.hub.load(
            'facebookresearch/dinov2', "dinov2_vitg14"
        ).to(device)


        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.device = device

        self.transform_clip = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

        self.transform_dino = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
        )

        self.decoder = LinearLayer()

        sam = sam_hq_model_registry["vit_h"]("src/eval/submission/ckpts/sam_hq_vit_h.pth").to(device)
        self.sam_predictor = SamPredictor(sam)


    def grounding_segmentation(self, image_path, mask_path, grounding_config):
        os.makedirs(f"{mask_path}",exist_ok=True)

        image_pil, _ = load_image(image_path)

        grounding_output = infer_dino(
            image_pil,
            grounding_config["text_prompt"],
            grounding_config["box_threshold"],
            grounding_config["text_threshold"],
            grounding_config["area_threshold"],
            self.grounding_model,
            self.grounding_processor,
            self.device,
        )

        boxes_filt, pred_phrases, scores = grounding_output[0]["boxes"], grounding_output[0]["labels"], grounding_output[0]["scores"]

        for i in range(len(pred_phrases)):
            pred_phrases[i] = pred_phrases[i] + f"({scores[i]:.2f})"

        background_box = list()
        for i,text in enumerate(pred_phrases):
            for j in grounding_config['background_prompt'].split('.'):
                if j in text.replace(' - ','-') and j != ' ' and j != '':
                    background_box.append(i)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        boxes_filt = boxes_filt.cpu()

        if boxes_filt.size(0) == 0:

            if self.class_name == "pushpins":
                refined_masks = np.zeros((448,448), dtype=np.uint8)
            else:
                refined_masks = np.zeros((256,256), dtype=np.uint8)

            cv2.imwrite(f"{mask_path}/refined_masks.png",refined_masks)

            with open(f"{mask_path}/pred_phrases.json", 'w', encoding='utf-8') as f:
                json.dump(pred_phrases, f, ensure_ascii=False, indent=4)

        else:


            boxes_filt = boxes_filt.cpu()

            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )


            if len(background_box) != 0:
                backgrounds = torch.stack([masks[i] for i in background_box])
                background = torch.sum(backgrounds,dim=0).squeeze().cpu().numpy()
                background = np.where(background!=0,255,0).astype(np.uint8)
            else:
                background = np.zeros_like(masks[0][0].cpu().numpy()).astype(np.uint8)

            masks = torch.stack([masks[i] for i in range(len(masks)) if i not in background_box])
            masks = turn_binary_to_int(masks[:,0,:,:].cpu().numpy())

            masks = split_masks_by_connected_component(masks)

            if len(masks) == 0:
                if self.class_name == "pushpins":
                    masks = np.zeros((448,448), dtype=np.uint8)
                else:
                    masks = np.zeros((256,256), dtype=np.uint8)
            else:
                masks = filter_by_combine(masks)
                if self.class_name == "screw_bag":
                    masks = filter_masks_below_area(masks, 200)
                masks = merge_masks(masks, reverse=True)

            cv2.imwrite(f"{mask_path}/grounding_mask.png",masks)


            refined_masks = cv2.imread(f"{mask_path}/grounding_mask.png",cv2.IMREAD_GRAYSCALE)
            refined_masks, _ = split_masks_from_one_mask(refined_masks)
            refined_masks = post_process_masks(refined_masks, self.class_name)

            if len(refined_masks) == 0:
                if self.class_name == "pushpins":
                    refined_masks = np.zeros((448,448), dtype=np.uint8)
                else:
                    refined_masks = np.zeros((256,256), dtype=np.uint8)
            else:
                refined_masks = merge_masks(refined_masks)

            cv2.imwrite(f"{mask_path}/refined_masks.png",refined_masks)
            with open(f"{mask_path}/pred_phrases.json", 'w', encoding='utf-8') as f:
                json.dump(pred_phrases, f, ensure_ascii=False, indent=4)


    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:

        torch.cuda.empty_cache()
        few_shot_samples = setup_data.get("few_shot_samples")
        self.class_name = setup_data.get("dataset_category")
        self.shot = len(few_shot_samples)

        if self.class_name in ["breakfast_box", "juice_bottle"]:
            grounding_model_id = "IDEA-Research/grounding-dino-tiny"
        else:
            grounding_model_id = "IDEA-Research/grounding-dino-base"

        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        

        self.sampler = GreedyCoresetSampler(percentage= 0.2 / self.shot, device=self.device)
        
        if self.class_name == "screw_bag":
            few_shot_samples = rotate(few_shot_samples, 3, interpolation=InterpolationMode.BILINEAR)
            few_shot_samples = crop(few_shot_samples, 35, 23, 180, 175)

        self.grounding_config = {}
        self.grounding_config['background_prompt'] = ""
        if self.class_name == "breakfast_box":
            self.grounding_config['text_prompt'] = "almond . apple . container . oatmeal . orange . banana . box . peach . nut"
            self.grounding_config['box_threshold'] = 0.15
            self.grounding_config['text_threshold'] = 0.15
            self.grounding_config['area_threshold'] = 1
            
        elif self.class_name == "screw_bag":
            self.grounding_config['text_prompt'] = "bolt . attach . key . nut . screw . washer . tool . stick . wrench . silver"
            self.grounding_config['box_threshold'] = 0.21
            self.grounding_config['text_threshold'] = 0.21
            self.grounding_config['area_threshold'] = 0.2
        elif self.class_name == "splicing_connectors":
            self.grounding_config['text_prompt'] = "attach. cable. connector. hook. electric outlet. plug. pole. socket. wire"
            self.grounding_config['box_threshold'] = 0.2
            self.grounding_config['text_threshold'] = 0.2
            self.grounding_config['area_threshold'] = 1
        elif self.class_name == "pushpins":
            self.grounding_config['text_prompt'] = "pushpin . pin"
            self.grounding_config['box_threshold'] = 0.15
            self.grounding_config['text_threshold'] = 0.15
            self.grounding_config['area_threshold'] = 0.1
        elif self.class_name == "juice_bottle":
            self.grounding_config['text_prompt'] = "bottle . label . banana . cherry . orange"
            self.grounding_config['box_threshold'] = 0.25
            self.grounding_config['text_threshold'] = 0.25
            self.grounding_config['area_threshold'] = 1

        clip_transformed_normal_image = self.transform_clip(few_shot_samples).to(
            self.device
        )
        dino_transformed_normal_image = self.transform_dino(few_shot_samples).to(
            self.device
        )

        with torch.no_grad():
            _, self.normal_patch_tokens = (
                self.clip_model.encode_image(
                    clip_transformed_normal_image, self.out_layers
                )
            )

            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)

            for i in range(len(self.normal_patch_tokens)):
                self.normal_patch_tokens[i] = self.normal_patch_tokens[i].reshape(-1, 1024)
                self.normal_patch_tokens[i] = self.sampler.run(self.normal_patch_tokens[i])


            self.normal_dino_patches = self.dinov2_net.forward_features(
                dino_transformed_normal_image
            )["x_norm_patchtokens"]

            self.normal_dino_patches = self.normal_dino_patches.reshape(-1, 1536)
            self.normal_dino_patches = self.sampler.run(self.normal_dino_patches)

        self.part_num = {}
        self.part_num['breakfast_box'] = 6
        self.part_num['juice_bottle'] = 3
        self.part_num['screw_bag'] = 6
        self.part_num['splicing_connectors'] = 3
        self.part_num['pushpins'] = 15

        self.image_idx = 0


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
        batch_size = image.shape[0]
        print("batch_size:", batch_size)

        batch_pred_scores = []
        # batch_results = []

        for batch_idx in range(batch_size):
            self.image_idx += 1
            single_image = image[batch_idx:batch_idx+1]

            if self.class_name == "screw_bag":
                single_image = rotate(single_image, 3, interpolation=InterpolationMode.BILINEAR)
                single_image = crop(single_image, 35, 23, 180, 175)

            clip_transformed_image = self.transform_clip(single_image)
            dino_transformed_image = self.transform_dino(single_image)

            with torch.no_grad():
                image_features, patch_tokens = self.clip_model.encode_image(
                    clip_transformed_image, self.out_layers
                )

                image_features = image_features[:, 0, :]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_tokens = self.decoder(patch_tokens)
                dino_patch_tokens = self.dinov2_net.forward_features(
                    dino_transformed_image
                )["x_norm_patchtokens"]

                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:
                        continue
                    patch_tokens_reshaped = patch_tokens[i].view(
                        int((self.image_size / 14) ** 2), 1, 1024
                    )
                    normal_tokens_reshaped = self.normal_patch_tokens[i].reshape(1, -1, 1024)
                    cosine_similarity_matrix = F.cosine_similarity(
                        patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                    )

                    if self.class_name == "pushpins":
                        sim_max = cosine_similarity_matrix.topk(5, dim=1)[0].mean()
                    else:
                        sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims, dim=0), dim=0)
                anomaly_map_ret = 1 - sim

            
                if self.class_name in ["pushpins", "screw_bag"]:
                    anomaly_map_structure = anomaly_map_ret
                else:
                    dino_patch_tokens_reshaped = dino_patch_tokens.view(-1, 1, 1536)
                    dino_normal_tokens_reshaped = self.normal_dino_patches.reshape(1, -1, 1536)
                    cosine_similarity_matrix = F.cosine_similarity(
                        dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                    )
                    sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)

                    anomaly_map_ret_dino = 1 - sim_max_dino

                    anomaly_map_structure = anomaly_map_ret + anomaly_map_ret_dino

                structure_score = anomaly_map_structure.max().item()

            image_save_path = f"src/eval/submission/test_images/{self.class_name}/{str(self.image_idx)}/image_0.jpg"

            if not os.path.exists(image_save_path):

                if self.class_name == "pushpins":
                    image_to_save = F.interpolate(
                        single_image, size=(448, 448), mode="bilinear", align_corners=True
                    )
                else:
                    image_to_save = single_image
                    
                save_tensor_as_jpg(image_to_save, save_dir=f"src/eval/submission/test_images/{self.class_name}/{str(self.image_idx)}", class_name=self.class_name)
                self.grounding_segmentation(
                    image_save_path, f"src/eval/submission/test_masks/{self.class_name}/{str(self.image_idx)}", self.grounding_config
                )


            logical_score = 0
            masks, _ = split_masks_from_one_mask_sort(cv2.imread(f"src/eval/submission/test_masks/{self.class_name}/{str(self.image_idx)}/refined_masks.png", cv2.IMREAD_GRAYSCALE))



            if len(masks) != self.part_num[self.class_name]:
                logical_score += 1

            else:
                logical_score += compute_logical_score(masks, self.class_name, self.image_idx)
            
            final_score = 1 * logical_score + 1 * structure_score
            batch_pred_scores.append(final_score)
            
        pred_scores = torch.tensor(batch_pred_scores, device=image.device)       

        return ImageBatch(
            image = image,
            pred_score = pred_scores,
        )
    

