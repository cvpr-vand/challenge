"""Model for submission."""

import torch
from anomalib.data import ImageBatch
from torch import nn

import warnings
warnings.filterwarnings("ignore")
from torchvision.transforms import v2
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms.v2.functional import resize, crop, rotate, InterpolationMode
import torch.nn.functional as F
# import kornia as K
import numpy as np
import torchvision
from sklearn.cluster import KMeans
import cv2
from .modules import DinoFeaturizer, PromptLearner, Adapter, TextEncoder, LinearLayer_fc
import glob
import random

# from anomalib.models.image.winclip.torch_model import WinClipModel
# from .prompt_ensemble import encode_text_with_prompt_ensemble
from .utils.filter_algorithm import filter_bg_noise
from .utils.sampler import GreedyCoresetSampler
from .utils.crf import dense_crf
from . import qkv_open_clip as open_clip
import os
from .utils.utils_area import (
    get_area_list_new,
    get_area_only_histo,
    train_select_binary_offsets,
    test_select_binary_offsets,
)

prompt_normal = ["normal {}"]
prompt_abnormal = ["abnormal {}"]

all_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "screw_bag",
    "splicing_connectors",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]

gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)


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

        setup_seed(1024)

        clip_name = "ViT-L-14-336"
        self.image_size = 224
        pretrained = "openai"
        device = torch.device("cuda")
        self.out_layers = [6, 12, 18, 24]

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, self.image_size, pretrained=pretrained
        )  # CLIP

        self.dino_net = DinoFeaturizer()

        # self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # self.sam.to(device=device)

        # self.predictor = SamPredictor(self.sam)

        self.clip_model.to(device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.device = device

        # # NOTE: Create your transformation pipeline (if needed).
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

        self.just_resize = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
            ],
        )

        self.decoder = LinearLayer_fc(1024, 1024, 4, self.clip_model)
        self.adapter = Adapter(768)

        self.normal_prompt_learner = PromptLearner(
            all_obj_list,
            prompt_normal,
            self.clip_model,
            self.tokenizer,
            device,
            768,
            12,
        ).to(device)

        self.abnormal_prompt_learner = PromptLearner(
            all_obj_list,
            prompt_abnormal,
            self.clip_model,
            self.tokenizer,
            device,
            768,
            12,
        ).to(device)

        self.text_encoder = TextEncoder(self.clip_model).to(device)
        self.text_encoder.eval()


    def setup(self, setup_data: dict[str, torch.Tensor]) -> None:
        """Setup the model.

        Optional: Use this to pass few-shot images and dataset category to the model.

        Args:
            setup_data (dict[str, torch.Tensor]): The setup data.
        """
        # pass

        few_shot_samples = setup_data.get("few_shot_samples")
        self.class_name = setup_data.get("dataset_category")

        if self.class_name == "screw_bag":
            few_shot_samples = rotate(
                few_shot_samples, 3, interpolation=InterpolationMode.BILINEAR
            )
            few_shot_samples = crop(few_shot_samples, 35, 23, 180, 175)
            few_shot_samples = self.just_resize(few_shot_samples)

        self.shot = len(few_shot_samples)

        clip_transformed_normal_image = self.transform_clip(few_shot_samples).cuda()
        dino_transformed_normal_image = self.transform_dino(few_shot_samples).cuda()
        only_resized_normal_image = self.just_resize(few_shot_samples)

        if self.class_name == "screw_bag":
            num_cluster = 5
        elif self.class_name == "juice_bottle":
            num_cluster = 4
        elif self.class_name == "pushpins":
            num_cluster = 4
        else:
            num_cluster = 5

        self.part_num = {
            "breakfast_box": [4],
            "screw_bag": [2],
            "splicing_connectors": [2],
            "pushpins": [3],
            "juice_bottle": [3],
        }

        color_list = [
            [127, 123, 229],
            [195, 240, 251],
            [146, 223, 255],
            [243, 241, 230],
            [224, 190, 144],
            [178, 116, 75],
        ]
        color_tensor = torch.tensor(color_list)
        color_tensor = color_tensor[:, :, None, None]
        self.color_tensor = color_tensor.repeat(1, 1, self.image_size, self.image_size)

        # segment normal images
        train_feature_list = []
        part_num = -1
        area_yes = 1
        component_yes = 1
        while (
            part_num not in self.part_num[self.class_name]
            or not area_yes
            or not component_yes
        ):
            if self.class_name == "pushpins":
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.5, device="cuda"
                )
            elif self.class_name == "screw_bag":
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.5, device="cuda"
                )
            elif self.class_name == "breakfast_box":
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.5, device="cuda"
                )
            elif self.class_name == "juice_bottle":
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.5, device="cuda"
                )
            else:
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.01, device="cuda"
                )
            for Img in dino_transformed_normal_image:
                Img = Img.unsqueeze(0)
                feats0, f_lowdim = self.dino_net(Img)
                feats = feats0.squeeze()
                feats = feats.reshape(feats0.shape[1], -1).permute(1, 0)
                feats_sample = greedsampler_perimg.run(feats)
                train_feature_list.append(feats_sample)

            train_features = torch.cat(train_feature_list, dim=0)
            train_features = F.normalize(train_features, dim=1)
            # torch.save(train_features.cpu(), f"{self.class_name}.pth")
            train_features = train_features.cpu().numpy()
            kmeans = KMeans(init="k-means++", n_clusters=num_cluster)
            c = kmeans.fit(train_features)
            cluster_centers = torch.from_numpy(c.cluster_centers_)
            # torch.save(cluster_centers.cpu(), f"{self.class_name}_k{num_cluster}.pth")
            train_features_sampled = cluster_centers.cuda()
            train_features_sampled = train_features_sampled.unsqueeze(0).unsqueeze(0)
            self.train_features_sampled = train_features_sampled.permute(0, 3, 1, 2)

            for i, Img in enumerate(dino_transformed_normal_image):
                Img = Img.unsqueeze(0)
                # print(Img.shape)
                heatmap, heatmap_intra = get_heatmaps(
                    Img, self.train_features_sampled, self.dino_net, self.color_tensor
                )
                savepath = f"./{self.class_name}_heat/train/{i}"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                see_image(Img, heatmap, f"{savepath}", heatmap_intra)

            self.subdict = {}
            self.subdict[f"{self.class_name}"] = filter_bg_noise(".", self.class_name)

            # print(self.subdict)
            part_num = len(self.subdict[f"{self.class_name}"])

            train_file_path = f"./{self.class_name}_heat/train"
            trainfiles = sorted(
                glob.glob(train_file_path + "/*"), key=lambda x: int(x.split("/")[-1])
            )[: self.shot]

            if self.class_name == "splicing_connectors":
                areas = []
                for sub in self.subdict["splicing_connectors"]:
                    area_train, train_mean, train_std, k_offset = (
                        train_select_binary_offsets(trainfiles, sub)
                    )
                    areas.append(train_mean)

                if len(areas) == 2:
                    ratio = (areas[0] / areas[1]).item()
                else:
                    ratio = 1

                if ratio > 2 or ratio < 1 / 2:
                    area_yes = 1
                else:
                    area_yes = 0

        with torch.no_grad():
            self.normal_image_features, self.normal_patch_tokens = (
                self.clip_model.encode_image(
                    clip_transformed_normal_image, self.out_layers
                )
            )

            self.normal_dino_patches, _ = self.dino_net(dino_transformed_normal_image)
            # self.normal_sam_patches = []
            # for i in range(len(only_resized_normal_image)):
            #     self.predictor.set_image(only_resized_normal_image[i])
            #     sam_feature = self.predictor.get_image_embedding()
            #     sam_feature = F.interpolate(
            #         sam_feature, size=16, mode="bilinear", align_corners=True
            #     )
            #     self.normal_sam_patches.append(sam_feature)

            # self.normal_sam_patches = torch.cat(self.normal_sam_patches, dim=0)
            self.normal_dino_patches = F.interpolate(
                self.normal_dino_patches, size=16, mode="bilinear", align_corners=True
            )
            self.normal_dino_patches = (
                self.normal_dino_patches.transpose(0, 1)
                .reshape(384, -1)
                .transpose(0, 1)
            )
            # self.normal_image_features = self.normal_image_features[:, 0, :]

            if self.class_name == "juice_bottle":
                self.normal_image_features = self.adapter(self.normal_image_features)

            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)

            # self.text_features = encode_text_with_prompt_ensemble(self.clip_model, [self.class_name], self.tokenizer, self.device)

            self.selected_features_dino = []
            self.part_normal_patch_tokens_dino = []

            # for layer in range(len(self.normal_patch_tokens)):
            # selected_features = []
            normal_dino_patches = self.normal_dino_patches.reshape(
                self.shot, 384, 256
            ).transpose(-2, -1)
            for sub in self.subdict[f"{self.class_name}"]:
                train_files = [
                    f"./{self.class_name}_heat/train/{i}/heatresult{sub}.jpg"
                    for i in range(len(few_shot_samples))
                ]
                sub_features = []
                sub_patch_features = []
                for j in range(len(train_files)):
                    normal_mask_path = train_files[j]
                    normal_mask = cv2.imread(normal_mask_path, 0)
                    gray_cal_otsu = normal_mask[
                        10 : normal_mask.shape[0] - 10, 10 : normal_mask.shape[0] - 10
                    ]
                    ret0, thresh0 = cv2.threshold(
                        gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    ret, thresh = cv2.threshold(normal_mask, ret0, 1, cv2.THRESH_BINARY)

                    thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
                    thresh = F.interpolate(
                        thresh, size=16, mode="bilinear", align_corners=True
                    ).reshape(256)
                    selected_feature = normal_dino_patches[j][thresh]
                    sub_patch_features.append(selected_feature)
                    sub_features.append(selected_feature.mean(dim=0))

                sub_patch_features = torch.cat(sub_patch_features, dim=0)
                sub_features = torch.stack(sub_features, dim=0)
                # selected_features.append(sub_features)

                sub_features = sub_features / sub_features.norm()
                # print(sub_features.shape)
                # print(sub_patch_features.shape)
                self.part_normal_patch_tokens_dino.append(sub_patch_features)
                self.selected_features_dino.append(sub_features)

            self.selected_features = []
            self.part_normal_patch_tokens = [[], [], [], []]

            for layer in range(len(self.normal_patch_tokens)):
                selected_features = []
                for sub in self.subdict[f"{self.class_name}"]:
                    train_files = [
                        f"./{self.class_name}_heat/train/{i}/heatresult{sub}.jpg"
                        for i in range(len(few_shot_samples))
                    ]
                    sub_features = []
                    sub_patch_features = []
                    for j in range(len(train_files)):
                        normal_mask_path = train_files[j]
                        normal_mask = cv2.imread(normal_mask_path, 0)
                        gray_cal_otsu = normal_mask[
                            10 : normal_mask.shape[0] - 10,
                            10 : normal_mask.shape[0] - 10,
                        ]
                        ret0, thresh0 = cv2.threshold(
                            gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        ret, thresh = cv2.threshold(
                            normal_mask, ret0, 1, cv2.THRESH_BINARY
                        )

                        thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
                        thresh = F.interpolate(
                            thresh, size=16, mode="bilinear", align_corners=True
                        ).reshape(256)
                        selected_feature = self.normal_patch_tokens[layer][j][thresh]
                        sub_patch_features.append(selected_feature)
                        sub_features.append(selected_feature.mean(dim=0))

                    sub_patch_features = torch.cat(sub_patch_features, dim=0)
                    sub_features = torch.stack(sub_features, dim=0)

                    sub_features = sub_features / sub_features.norm()
                    selected_features.append(sub_features)
                    # print(sub_features.shape)
                    # print(sub_patch_features.shape)
                    self.part_normal_patch_tokens[layer].append(sub_patch_features)
                self.selected_features.append(selected_features)


            if self.class_name == "pushpins":

                component_min = 100000
                component_min_idx = -1
                for idx in range(len(self.subdict["pushpins"])):
                    sub = self.subdict["pushpins"][idx]
                    area_train, train_mean, train_std, k_offset = (
                        train_select_binary_offsets(trainfiles, sub)
                    )
                    componets_num = len(get_area_list_new(trainfiles, sub, k_offset))
                    if componets_num < component_min:
                        component_min = componets_num
                        component_min_idx = idx
                del self.subdict["pushpins"][component_min_idx]

                area_min = 100000
                area_min_idx = -1
                for sub in self.subdict["pushpins"]:
                    area_train, train_mean, train_std, k_offset = (
                        train_select_binary_offsets(trainfiles, sub)
                    )
                    # componets_num = len(get_area_list_new(trainfiles, sub, k_offset))
                    if train_mean < area_min:
                        area_min = train_mean
                        area_min_idx = sub
                self.subdict["pushpins"] = [area_min_idx]

                if (
                    componets_num >= 15 * self.shot - self.shot // 4
                    and componets_num <= 15 * self.shot + self.shot // 4
                ):
                    component_yes = 1
                else:
                    component_yes = 0
            # print(self.subdict)


            # print(self.subdict)

            area_train_all = []
            color_train_all = []

            # component
            # testhis = []
            # trainhis = []

            # train_file_path = f'./{self.class_name}_heat/train'
            # trainfiles = sorted(glob.glob(train_file_path+'/*'), key=lambda x: int(x.split('/')[-1]))[:self.shot]

            self.k_offsets = {}

            for sub in self.subdict[f"{self.class_name}"]:
                area_train, self.train_mean, self.train_std, k_offset = (
                    train_select_binary_offsets(trainfiles, sub)
                )
                # print()
                self.k_offsets[sub] = k_offset
                area_train_all.append(area_train)
                # train component
                component_area_list = get_area_list_new(trainfiles, sub, k_offset)

                # print(len(component_area_list), self.train_mean)
                component_area = np.asarray(component_area_list)
                component_area_mean = np.mean(component_area)
                dbscan_r = component_area_mean * 0.1
                dbscan_min = 10
                self.dbscan = DBSCAN(eps=dbscan_r, min_samples=dbscan_min)

                self.dbscan.fit(component_area)
                self.nn_connection = NearestNeighbors(n_neighbors=1)
                self.nn_connection.fit(component_area)
                self.train_histo_numpy = get_area_only_histo(
                    trainfiles, sub, k_offset, self.dbscan, self.nn_connection
                )
                # print(train_histo_numpy)
                self.nn_connection_histo = NearestNeighbors(n_neighbors=1)
                self.nn_connection_histo.fit(self.train_histo_numpy)

                # train_color
                _, color_train, self.cmean, self.cstd = test_select_binary_offsets(
                    trainfiles, sub, k_offset, self.train_mean, self.train_std, 0, 0
                )
                color_train_all.append(color_train)

            area_train_all_numpy = np.concatenate(area_train_all, axis=1)
            color_train_all_numpy = np.concatenate(color_train_all, axis=1)
            train_global = np.concatenate(
                (area_train_all_numpy, color_train_all_numpy), axis=1
            )
            self.nn_train_global = NearestNeighbors(n_neighbors=1)
            self.nn_train_global.fit(train_global)

        # normal_prompts = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        # normal_prompts = [x.format(self.class_name) for x in normal_prompts]

        # abnormal_prompts = detailed_descriptions[]



    def weights_url(self, category: str) -> str | None:
        """URL to the model weights.

        You can optionally use the category to download specific weights for each category.
        """
        # TODO: Implement this if you want to download the weights from a URL
        # return None

        # urls = {
        #     "breakfast_box": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/breakfast_box.pth",
        #     "juice_bottle": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/juice_bottle.pth",
        #     "pushpins": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/pushpins.pth",
        #     "screw_bag": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/screw_bag.pth",
        #     "splicing_connectors": "https://github.com/ashwinvaidya17/cvpr-2025-challenge/releases/download/sample-weights-model/splicing_connectors.pth",
        # }
        # return urls[category]

        single_url = "https://github.com/liuzhen19/cvpr-vand-challenge/releases/download/v0.1.0/checkpoint.pt"
        return single_url


    def forward(self, image: torch.Tensor) -> ImageBatch:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            ImageBatch: The output image batch.
        """
        # # TODO: Implement the forward pass of the model.
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
        # batch = self.just_resize(batch)
        if self.class_name == "screw_bag":
            batch = rotate(batch, 3, interpolation=InterpolationMode.BILINEAR)
            batch = crop(batch, 35, 23, 180, 175)
            batch = self.just_resize(batch)

        clip_transformed_image = self.transform_clip(batch)
        dino_transformed_image = self.transform_dino(batch)
        only_resized_image = self.just_resize(batch)

        heatmap, heatmap_intra = get_heatmaps(
            dino_transformed_image,
            self.train_features_sampled,
            self.dino_net,
            self.color_tensor,
        )
        savepath = f"./{self.class_name}_heat/test/0"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        see_image(dino_transformed_image, heatmap, f"{savepath}", heatmap_intra)

        image_features, patch_tokens = self.clip_model.encode_image(
            clip_transformed_image, self.out_layers
        )
        dino_patch_tokens, _ = self.dino_net(dino_transformed_image)
        dino_patch_tokens = F.interpolate(
            dino_patch_tokens, size=16, mode="bilinear", align_corners=True
        )

        # self.predictor.set_image(batch[0])
        # sam_patch_tokens = self.predictor.get_image_embedding()
        # sam_patch_tokens = F.interpolate(
        #     sam_patch_tokens, size=16, mode="bilinear", align_corners=True
        # )

        if self.class_name == "juice_bottle":
            image_features = self.adapter(image_features)

        image_features = image_features / image_features.norm()

        self.normal_image_features = (
            self.normal_image_features / self.normal_image_features.norm()
        )

        global_score = (
            1
            - (image_features @ self.normal_image_features.transpose(-2, -1))
            .max()
            .item()
        )

        normal_prompts = self.normal_prompt_learner(image_features, self.class_name)
        normal_tokenized_prompts = self.normal_prompt_learner.tokenized_prompts[
            self.class_name
        ].to(self.device)

        abnormal_prompts = self.abnormal_prompt_learner(image_features, self.class_name)
        abnormal_tokenized_prompts = self.abnormal_prompt_learner.tokenized_prompts[
            self.class_name
        ].to(self.device)

        normal_text_features = self.text_encoder(
            normal_prompts[0], normal_tokenized_prompts
        )
        normal_text_features = normal_text_features / normal_text_features.norm(
            dim=-1, keepdim=True
        )
        abnormal_text_features = self.text_encoder(
            abnormal_prompts[0], abnormal_tokenized_prompts
        )
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
            dim=-1, keepdim=True
        )

        normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
        normal_text_features = normal_text_features / normal_text_features.norm()
        normal_text_features = normal_text_features.unsqueeze(1)
        abnormal_text_features = abnormal_text_features.mean(dim=0, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm()
        abnormal_text_features = abnormal_text_features.unsqueeze(1)
        text_features = torch.cat(
            [normal_text_features, abnormal_text_features], dim=1
        ).to(self.device)

        logits = torch.softmax(
            image_features.unsqueeze(1) @ text_features.transpose(-2, -1), dim=-1
        )

        patch_tokens = self.decoder(patch_tokens)
        sims = []
        for i in range(len(patch_tokens)):
            # patch_tokens[i] = patch_tokens[i][:,1:,:]
            patch_tokens_reshaped = patch_tokens[i].view(
                int((self.image_size / 14) ** 2), 1, 1024
            )
            normal_tokens_reshaped = self.normal_patch_tokens[i].reshape(1, -1, 1024)
            cosine_similarity_matrix = F.cosine_similarity(
                patch_tokens_reshaped, normal_tokens_reshaped, dim=2
            )
            sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
            sims.append(sim_max)

        sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
        sim = F.interpolate(sim, size=256, mode="bilinear", align_corners=True)
        anomaly_map_ret = 1 - sim

        # image_path_split = image_path.split("/")

        # sims = []
        # for layer in range(len(self.normal_patch_tokens)):
        #     sim_layer = []
        #     for i in range(len(self.subdict[f'{self.class_name}'])):
        #         sub = self.subdict[f'{self.class_name}'][i]
        #         heat_path = f"./{self.class_name}_heat/test/0/heatresult{sub}.jpg"
        #         heat_mask = cv2.imread(heat_path, 0)
        #         gray_cal_otsu = heat_mask[
        #             10 : heat_mask.shape[0] - 10, 10 : heat_mask.shape[0] - 10
        #         ]
        #         ret0, thresh0 = cv2.threshold(
        #             gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        #         )
        #         ret, thresh = cv2.threshold(heat_mask, ret0, 1, cv2.THRESH_BINARY)

        #         thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
        #         thresh = F.interpolate(thresh,size=28, mode='bilinear', align_corners=True).reshape(256)
        #         if thresh.sum() < 1:
        #             continue

        #         # selected_features = patch_tokens[layer][0][thresh].view(-1, 1, 1024) #[k, 1, 1024]
        #         selected_features = dino_patch_tokens.view(384, -1).transpose(0,1)[thresh].view(-1, 1, 384)
        #         normal_tokens_reshaped = self.part_normal_patch_tokens_dino[i].reshape(1, -1, 384)

        #         cosine_similarity_matrix = F.cosine_similarity(selected_features, normal_tokens_reshaped, dim=2)
        #         sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
        #         sim_layer.append(sim_max)

        #     sim_layer = torch.cat(sim_layer,dim=0)
        #     # print(sim_layer.shape)
        #     sims.append(sim_layer)

        # anomaly_map_ret_part_dino = 1 - torch.mean(torch.stack(sims,dim=0), dim=0)

        dino_patch_tokens_reshaped = dino_patch_tokens.view(1, 384, -1).permute(
            (2, 0, 1)
        )
        dino_normal_tokens_reshaped = self.normal_dino_patches.reshape(1, -1, 384)
        cosine_similarity_matrix = F.cosine_similarity(
            dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
        )
        sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
        sim_max_dino = sim_max_dino.reshape(1, 1, 16, 16)
        sim_max_dino = F.interpolate(
            sim_max_dino, size=256, mode="bilinear", align_corners=True
        )
        anomaly_map_ret_dino = 1 - sim_max_dino

        # sam_patch_tokens_reshaped = sam_patch_tokens.view(1, 256, -1).permute((2, 0, 1))
        # sam_normal_tokens_reshaped = self.normal_sam_patches.reshape(1, -1, 256)
        # cosine_similarity_matrix = F.cosine_similarity(
        #     sam_patch_tokens_reshaped, sam_normal_tokens_reshaped, dim=2
        # )
        # sim_max_sam, _ = torch.max(cosine_similarity_matrix, dim=1)
        # sim_max_sam = sim_max_sam.reshape(1, 1, 16, 16)
        # sim_max_sam = F.interpolate(
        #     sim_max_sam, size=256, mode="bilinear", align_corners=True
        # )
        # anomaly_map_ret_sam = 1 - sim_max_sam
        # print(anomaly_map_ret_sam.max().item())

        dists_multi_layers_dino = []
        for layer in [-1]:  # range(len(self.normal_patch_tokens)):
            dists = []
            for i in range(len(self.subdict[f"{self.class_name}"])):
                sub = self.subdict[f"{self.class_name}"][i]
                heat_path = f"./{self.class_name}_heat/test/0/heatresult{sub}.jpg"
                heat_mask = cv2.imread(heat_path, 0)
                gray_cal_otsu = heat_mask[
                    10 : heat_mask.shape[0] - 10, 10 : heat_mask.shape[0] - 10
                ]
                ret0, thresh0 = cv2.threshold(
                    gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                ret, thresh = cv2.threshold(heat_mask, ret0, 1, cv2.THRESH_BINARY)

                thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
                thresh = F.interpolate(
                    thresh, size=16, mode="bilinear", align_corners=True
                ).reshape(256)

                if thresh.sum() < 1:
                    continue

                selected_feature = (
                    dino_patch_tokens.view(384, -1)
                    .transpose(0, 1)[thresh]
                    .mean(dim=0, keepdim=True)
                )
                selected_feature = selected_feature / selected_feature.norm()
                dist = (
                    1
                    - (selected_feature @ self.selected_features_dino[i].T).max().item()
                )
                dists.append(dist)
            dists = torch.tensor(dists)
            dists_multi_layers_dino.append(dists)
        dists_multi_layers_dino = torch.mean(
            torch.stack(dists_multi_layers_dino, dim=0), dim=0
        )

        sims = []
        for layer in range(len(self.normal_patch_tokens)):
            sim_layer = []
            for i in range(len(self.subdict[f"{self.class_name}"])):
                sub = self.subdict[f"{self.class_name}"][i]
                heat_path = f"./{self.class_name}_heat/test/0/heatresult{sub}.jpg"
                heat_mask = cv2.imread(heat_path, 0)
                gray_cal_otsu = heat_mask[
                    10 : heat_mask.shape[0] - 10, 10 : heat_mask.shape[0] - 10
                ]
                ret0, thresh0 = cv2.threshold(
                    gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                ret, thresh = cv2.threshold(heat_mask, ret0, 1, cv2.THRESH_BINARY)

                thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
                thresh = F.interpolate(
                    thresh, size=16, mode="bilinear", align_corners=True
                ).reshape(256)
                if thresh.sum() < 1:
                    continue

                selected_features = patch_tokens[layer][0][thresh].view(
                    -1, 1, 1024
                )  # [k, 1, 1024]
                normal_tokens_reshaped = self.part_normal_patch_tokens[layer][
                    i
                ].reshape(
                    1, -1, 1024
                )  # []

                cosine_similarity_matrix = F.cosine_similarity(
                    selected_features, normal_tokens_reshaped, dim=2
                )
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                sim_layer.append(sim_max)

            sim_layer = torch.cat(sim_layer, dim=0)
            # print(sim_layer.shape)
            sims.append(sim_layer)

        anomaly_map_ret_part = 1 - torch.mean(torch.stack(sims, dim=0), dim=0)

        dists_multi_layers = []
        for layer in [-1]:  # range(len(self.normal_patch_tokens)):
            dists = []
            for i in range(len(self.subdict[f"{self.class_name}"])):
                sub = self.subdict[f"{self.class_name}"][i]
                heat_path = f"./{self.class_name}_heat/test/0/heatresult{sub}.jpg"
                heat_mask = cv2.imread(heat_path, 0)
                gray_cal_otsu = heat_mask[
                    10 : heat_mask.shape[0] - 10, 10 : heat_mask.shape[0] - 10
                ]
                ret0, thresh0 = cv2.threshold(
                    gray_cal_otsu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                ret, thresh = cv2.threshold(heat_mask, ret0, 1, cv2.THRESH_BINARY)

                thresh = torch.tensor(thresh).reshape(1, 1, 224, 224)
                thresh = F.interpolate(
                    thresh, size=16, mode="bilinear", align_corners=True
                ).reshape(256)

                if thresh.sum() < 1:
                    continue

                selected_feature = patch_tokens[layer][0][thresh].mean(
                    dim=0, keepdim=True
                )
                selected_feature = selected_feature / selected_feature.norm()
                dist = (
                    1
                    - (selected_feature @ self.selected_features[layer][i].T)
                    .max()
                    .item()
                )
                dists.append(dist)
            dists = torch.tensor(dists)
            dists_multi_layers.append(dists)
        dists_multi_layers = torch.mean(torch.stack(dists_multi_layers, dim=0), dim=0)

        test_file_path = f"./{self.class_name}_heat/test"
        testfiles = sorted(
            glob.glob(test_file_path + "/*"), key=lambda x: int(x.split("/")[-1])
        )[: self.shot]

        area_test_all = []
        color_test_all = []
        score_test_all = 0

        for sub in self.subdict[f"{self.class_name}"]:

            area_test_all, color_test_all, test_histo_numpy = test_area_color_component(
                area_test_all,
                color_test_all,
                testfiles,
                sub,
                self.k_offsets[sub],
                self.train_mean,
                self.train_std,
                self.cmean,
                self.cstd,
                self.dbscan,
                self.nn_connection,
            )

            # trainhis.append(train_histo_numpy)
            # compont test error
            alpha = 0.5
            dis_test, indices = self.nn_connection_histo.kneighbors(test_histo_numpy)
            dis_test = (
                np.mean(dis_test, axis=1)
                * alpha
                / (self.train_histo_numpy.shape[1] * self.train_histo_numpy.shape[1])
            )
            # print(indices)
            score_test_all = score_test_all + dis_test
            # print(score_test_all)
            # testhis.append(test_histo_numpy)

        score_test_all = test_global_info(
            score_test_all, area_test_all, color_test_all, self.nn_train_global
        )
        score_test_all = score_test_all.item()
        if score_test_all > 1:
            score_test_all = 1
        if score_test_all < 0:
            score_test_all = 0

        # print(score_test_all)
        score_dict = {}
        if self.class_name == "juice_bottle":
            score_dict = {
                "pred_score": torch.tensor((
                    10 * anomaly_map_ret.max().item()
                    + 5 * anomaly_map_ret_dino.max().item()
                )
                + 25 * global_score
                + 20 * logits[0, 0, 1].item())
            }
        elif self.class_name == "pushpins":
            score_dict = {
                "pred_score":  torch.tensor((10 * anomaly_map_ret.max().item())
                + 5 * global_score
                + (50 * score_test_all))
            }
        elif self.class_name == "screw_bag":
            score_dict =  {
                "pred_score":  torch.tensor((
                    10 * anomaly_map_ret.max().item()
                    + 20 * anomaly_map_ret_dino.max().item()
                )
                + 10 * global_score
                + (
                    1 * score_test_all
                    + 100 * dists_multi_layers.mean().item()
                    + 5 * dists_multi_layers_dino.mean().item()
                ))
            }
        elif self.class_name == "splicing_connectors":
            score_dict = {
                "pred_score":  torch.tensor((
                    10 * anomaly_map_ret.max().item()
                    + 10 * anomaly_map_ret_dino.max().item()
                )
                + 30 * global_score
                + 1 * logits[0, 0, 1].item()
                + (
                    5 * dists_multi_layers.mean().item()
                    + 1 * dists_multi_layers_dino.mean().item()
                    + 1 * anomaly_map_ret_part.max().item()
                ))
            }
        else:  # "breakfast_box"
            score_dict = {
                "pred_score":  torch.tensor((
                    10 * anomaly_map_ret.max().item()
                    + 30 * anomaly_map_ret_dino.max().item()
                )
                + 30 * global_score
                + 1 * logits[0, 0, 1].item()
                + (
                    200 * dists_multi_layers.max().item()
                    + 100 * dists_multi_layers_dino.max().item()
                    + 10 * anomaly_map_ret_part.max().item()
                ))
            }
        
        return ImageBatch(
            image=image,
            pred_score=score_dict["pred_score"].to(image.device),
        )


def get_heatmaps(img, query_feature, net, color_tensor):
    with torch.no_grad():
        feats1, f1_lowdim = net(img.cuda())
    sfeats1 = query_feature
    attn_intra = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1)
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)
    heatmap_intra = (
        F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True)
        .squeeze(0)
        .detach()
        .cpu()
    )
    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf, heatmap_intra)
    heatmap_intra = torch.from_numpy(crf_result)
    d = heatmap_intra.argmax(dim=0)
    d = d[None, None, :, :]
    d = d.repeat(1, 3, 1, 1)
    seg_map = torch.zeros([1, 3, d.shape[2], d.shape[3]], dtype=torch.int64)
    for color in range(query_feature.shape[3]):
        seg_map = torch.where(d == color, color_tensor[color], seg_map)
    return seg_map, heatmap_intra


def see_image(data, heatmap, savepath, heatmap_intra):
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = np.clip((data * i_std + i_m) * 255, 0, 255).astype(np.uint8)
    data = data.transpose(1, 2, 0)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{savepath}/img.jpg", data)

    heatmap = heatmap[0, :, :, :].cpu().numpy()
    heatmap = heatmap.transpose(1, 2, 0)
    cv2.imwrite(f"{savepath}/heatresult.jpg", heatmap)

    for i in range(heatmap_intra.shape[0]):
        heat = heatmap_intra[i, :, :].cpu().numpy()
        heat = np.round(heat * 128).astype(np.uint8)
        cv2.imwrite(f"{savepath}/heatresult{i}.jpg", heat)


def test_area_color_component(
    area_test_all,
    color_test_all,
    files,
    sub,
    k_offset,
    train_mean,
    train_std,
    cmean,
    cstd,
    dbscan,
    nn_connection,
):
    area_test, color_test, _, _ = test_select_binary_offsets(
        files, sub, k_offset, train_mean, train_std, cmean, cstd
    )
    area_test_all.append(area_test)
    color_test_all.append(color_test)
    # test_good_component
    test_histo_numpy = get_area_only_histo(files, sub, k_offset, dbscan, nn_connection)
    return area_test_all, color_test_all, test_histo_numpy


def test_global_info(score, area_test, color_test, nn_train_global):
    area_test_numpy = np.concatenate(area_test, axis=1)
    color_test_numpy = np.concatenate(color_test, axis=1)
    test_global = np.concatenate((area_test_numpy, color_test_numpy), axis=1)
    dis_test_global, _ = nn_train_global.kneighbors(test_global)
    dis_test_global = np.mean(dis_test_global, axis=1)
    score_test = score + dis_test_global
    return score_test