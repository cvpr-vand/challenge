import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from eval.submission.gdino_sam2.sam2.build_sam import build_sam2
from eval.submission.gdino_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.path.append('/home/user/actions-runner/_work/challenge/challenge/src/eval/submission/gdino_sam2/grounding_dino')
#from eval.submission.gdino_sam2.grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.inference import load_model, load_image, predict
from torchvision import transforms as T
from PIL import Image


class GSAM2Predictor:
    def __init__(self, ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        SAM2_CHECKPOINT = os.path.join(current_dir, "checkpoints/sam2.1_hiera_large.pt")
        SAM2_MODEL_CONFIG =  "configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = os.path.join(current_dir, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT = os.path.join(current_dir, "gdino_checkpoints/groundingdino_swint_ogc.pth")

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

        self.device = DEVICE
        self.box_threshold = {
            "breakfast_box": 0.30,
            "juice_bottle": 0.3,   
            "pushpins": 0.15,
            "screw_bag": 0.15,
            "splicing_connectors": 0.23,
        }
        self.text_threshold = {
            "breakfast_box": 0.2,
            "juice_bottle": 0.25,
            "pushpins": 0.15,
            "screw_bag": 0.15,
            "splicing_connectors": 0.25,
        }

        self.text_prompt = {
            "breakfast_box": "orange. peach. oatmeal. banana chips. almonds. white box.",
            "juice_bottle": "glass bottle. cherry. orange. banana. label. tag.",
            "pushpins": "tools.",
            "screw_bag": "metal circle. long bolts. bolt.  circle. plastic bag.",
            "splicing_connectors": "connector.",
        }
        self.visualize = False
        self.orig_wh = {
            "breakfast_box": (1600, 1280),
            "juice_bottle": (800, 1600),
            "pushpins": (1700, 1000),
            "screw_bag": (1600, 1100),
            "splicing_connectors": (1700, 850),
        }
        self.img_size = 448

    def resize_to_original_aspect(self, img, original_size, max_size=256):
        W_orig, H_orig = original_size
        aspect_ratio = W_orig / H_orig  # 原始宽高比
        
        if aspect_ratio > 1:  # 原图是宽图（W > H）
            new_W = max_size
            new_H = int(max_size / aspect_ratio)
        else:  # 原图是高图（H > W）
            new_H = max_size
            new_W = int(max_size * aspect_ratio)
        
        return img.resize((new_W, new_H), Image.BILINEAR), (new_W, new_H)
    def compute_iou(self, box1, box2):
        # 计算iou矩阵
        # box1: (n, 4)
        # box2: (m, 4)
        # 计算交集
        x1 = np.maximum(box1[:, 0], box2[:, 0].reshape(-1, 1))
        y1 = np.maximum(box1[:, 1], box2[:, 1].reshape(-1, 1))
        x2 = np.minimum(box1[:, 2], box2[:, 2].reshape(-1, 1))
        y2 = np.minimum(box1[:, 3], box2[:, 3].reshape(-1, 1))
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        # 计算并集
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area.reshape(-1, 1) + box2_area - inter_area
        # 计算iou
        iou = inter_area / union_area
        # 将对角线设为0（避免自比较）
        np.fill_diagonal(iou, 0)
        return iou
    
    def cluster_mask(self, masks, boxes, obj_names, class_name):
        # Get configuration for target class
        target_names = self.objs_dict[class_name]  # Expected names for each object
        target_obj_num = self.foreground_num[class_name]  # Expected number of objects
        
        # Initialize cluster containers
        label_ids = [[] for _ in range(target_obj_num)]
        
        def find_largest_box(boxes):
            """Helper to find index of largest area box"""
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return np.argmax(areas)
        
        # Special handling for pushpins and splicing connectors
        if class_name in ["pushpins", "splicing_connectors"]:
            largest_idx = find_largest_box(boxes)
            label_ids[0].append(largest_idx)
            
            # Group remaining objects
            remaining_indices = [i for i in range(len(obj_names)) if i != largest_idx]
            if remaining_indices:
                if len(label_ids) > 1:
                    label_ids[1].extend(remaining_indices)
                else:
                    label_ids.append(remaining_indices)
            
        else:
            # General case: cluster by name matching
            
            for idx, obj in enumerate(obj_names):
                find_flag = False
                for j, expected_names in enumerate(target_names):
                    for name in expected_names:
                        if name in obj:
                            label_ids[j].append(idx)
                            find_flag = True
                            break
                    if find_flag:
                        break
        
        # Create clustered masks
        clustered = []
        for cluster_indices in label_ids:
            # if cluster_indices:  # Only process non-empty clusters
            combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
            for idx in cluster_indices:
                combined_mask = np.logical_or(combined_mask, masks[idx])
            clustered.append(combined_mask.astype(np.uint8))
        
        if class_name == "breakfast_box":
            # 比较麦片和香蕉皮的mask
            intersection = np.logical_and(clustered[-2], clustered[-1])
            area = np.sum(clustered[-1])
            if np.sum(intersection) > area * 0.5:
                clustered[-2][intersection] = 0
        
        return np.stack(clustered), label_ids

    def gdino_postprocess(self, boxes, confidences, labels, class_name):
        new_labels = labels
        delete_idx = []
        if class_name == "juice_bottle":
            # check labels
            tags = []
            fruits = []
            for idx, label in enumerate(labels):
                if "label" in label:
                    tags.append(idx)
                if label in ["banana", "cherry", "orange"]:
                    fruits.append(idx)
            num_lables = len(tags)
            num_fruit = len(fruits)
            
            if num_lables > 2 :
                # 移除置信度低的标签
                idx = tags[confidences[tags].argmin()]
                delete_idx.append(idx)

            if num_fruit > 1:
                if len(set([ labels[i] for i in fruits])) > 1:
                    # 移除置信度低的水果
                    idx = fruits[confidences[fruits].argmin()]
                    delete_idx.append(idx)

        elif class_name == "breakfast_box":
            # 003   # 
            chips = list()
            oatmeals = list()
            for idx, label in enumerate(labels):
                # if label in ["banana chips", "almonds", "banana chips almonds"]:
                if "banana" in label or "almonds" in label:
                    chips.append(idx)
            # 004 麦片多检测
                # if label in []:
                elif "oatmeal" in label:
                    oatmeals.append(idx)
            num_chips = len(chips)
            num_oatmeals = len(oatmeals)
            if num_chips > 1:
                idx = chips[confidences[chips].argmax()]
                delete_idx.extend([ i for i in chips if i != idx])
            
            if num_oatmeals > 1:
                idx = oatmeals[confidences[oatmeals].argmax()]
                delete_idx.extend([ i for i in oatmeals if i != idx])
            
            # 宽高过滤
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                if abs(x2-x1) < 50 or abs(y2-y1) < 50:
                    delete_idx.append(idx)
        elif class_name == "pushpins":
            # iou 过滤
            # 按照置信度排序
            sort_idx = np.argsort(confidences)[::-1]
            boxes = boxes[sort_idx]
            confidences = confidences[sort_idx]
            labels = [ labels[i] for i in sort_idx]
            new_labels = labels
            # 计算iou
            ious = self.compute_iou(boxes, boxes) 
            # 找出所有IOU>阈值的索引对
            rows, cols = np.where(ious > 0.2)
            
            # 确保每个对只出现一次 (i,j) 且 i < j
            duplicate_pairs = [(i, j) for i, j in zip(rows, cols) if i < j]

            for i, j in duplicate_pairs:
                if i not in delete_idx:
                    delete_idx.append(j)
            # 宽高过滤
            min_y1 = [0, 0, -1]     # y1, area, idx
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                if abs(x2-x1) < 15 or abs(y2-y1) < 15:
                    delete_idx.append(idx)
                
                if y1 < min_y1[0]:
                    min_y1[0] = y1
                    min_y1[1] = (x2-x1) * (y2-y1)
                    min_y1[2] = idx
            if min_y1[1] > 0 and min_y1[1] < 10000:
                delete_idx.append(min_y1[2])
        elif class_name == "screw_bag":
            sort_idx = np.argsort(confidences)[::-1]
            boxes = boxes[sort_idx]
            confidences = confidences[sort_idx]
            labels = [ labels[i] for i in sort_idx]
            new_labels = labels
            cicles = list()
            bolts = list()
            bags = list()
            for idx, label in enumerate(labels):
                if "circle" in label or "metal" in label:
                    cicles.append(idx)
                elif "bolt" in label:
                    bolts.append(idx)
                elif "bag" in label:
                    bags.append(idx)
            remain_bolts = list()
            remain_cicles = list()
            for idx, box, score in zip(cicles, boxes[cicles],confidences[cicles]):
                if max(abs(box[2]-box[0]), abs(box[3]-box[1])) > 80 :
                    delete_idx.append(idx)
                else:
                    remain_cicles.append(idx)

            for idx, box, score in zip(bolts, boxes[bolts],confidences[bolts]):
                if max(abs(box[2]-box[0]), abs(box[3]-box[1])) > 150 or (abs(box[2]-box[0])<50 and  abs(box[3]-box[1])< 50) :
                    delete_idx.append(idx)
                else:
                    remain_bolts.append(idx)

            ious = self.compute_iou(boxes, boxes)
            rows, cols = np.where(ious > 0.3)
            # 确保每个对只出现一次 (i,j) 且 i < j
            duplicate_pairs = [(i, j) for i, j in zip(rows, cols) if i < j]

            for i, j in duplicate_pairs:
                if i not in delete_idx:
                    if (i in remain_cicles and j in remain_cicles):
                        delete_idx.append(j)
                    if len(remain_bolts) > 3 and (i in remain_bolts and j in remain_bolts):
                        delete_idx.append(j)
            if len(bags) > 1: 
                # 移除置信度低的标签
                idx = bags[confidences[bags].argmin()]
                delete_idx.append(idx)

        if len(delete_idx) > 0:
            boxes = np.delete(boxes, delete_idx, axis=0)
            confidences = np.delete(confidences, delete_idx, axis=0)
            new_labels = [ labels[i] for i in range(len(labels)) if i not in delete_idx]
        return boxes, confidences, new_labels


    def forward(self, image_data, class_name, output_dir="outputs"):

        text = self.text_prompt[class_name]        
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        to_pil = T.ToPILImage()
        pil_img = to_pil(image_data)
        pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        sam_input = pil_img
        gdino_img_aspect,(gdino_w, gdino_h) = self.resize_to_original_aspect(pil_img, self.orig_wh[class_name], self.img_size)
        gdino_input = transform(gdino_img_aspect)
        self.sam2_predictor.set_image(sam_input)
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=gdino_input,
            caption=text,
            box_threshold=self.box_threshold[class_name],
            text_threshold=self.text_threshold[class_name],
        )
        w, h = gdino_w, gdino_h
        img_w, img_h = pil_img.size
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        input_boxes = input_boxes * np.array([img_w / w, img_h / h, img_w / w, img_h / h])
        confidences = confidences.numpy()
        input_boxes, confidences, labels = self.gdino_postprocess(
            boxes=input_boxes,
            confidences=confidences,
            labels=labels,
            class_name=class_name
        )
        #import pdb
        #pdb.set_trace()
        masks, scores, logits = self.sam2_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_boxes,
                                multimask_output=False,
                            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        class_names = labels

        if self.visualize:
            class_ids = np.array(list(range(len(class_names))))

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]
            img = np.array(sam_input)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids
            )

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

            clusted_masks, _ = self.cluster_mask(masks, boxes=input_boxes, obj_names=class_names, class_name=class_name)
            for idx, mask in enumerate(clusted_masks):
                cv2.imwrite(os.path.join(output_dir, f"mask_{idx}.png"), mask*255)
        return masks, input_boxes, class_names,pil_img
