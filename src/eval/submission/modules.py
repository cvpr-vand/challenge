import torch
from .utils.comad_utils import *
import torch.nn.functional as F
from .dino import vision_transformer as vits
from typing import Union, List, OrderedDict

class DinoFeaturizer(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.dim = 70 #dim
        patch_size = 8 #self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = 'feat'#self.cfg.dino_feat_type
        arch = 'vit_small' #self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)
        self.whetherdropout = False
        if arch == "vit_small" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        # print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url=url)
        self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        code = image_feat

        if self.whetherdropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code

class PromptLearner(nn.Module):
    def __init__(
        self, classnames, status, clip_model, tokenizer, device, dim, n_ctx
    ):
        super().__init__()
        n_ctx_1 = n_ctx
        vis_dim = dim
        ctx_dim = dim

        # 直接进行无初始化
        # print("Initializing a generic context")
        ctx_vectors_1 = torch.empty(n_ctx_1, ctx_dim)

        nn.init.normal_(ctx_vectors_1, std=0.02)
        prompt_part_1 =  " ".join(["X"] * n_ctx_1)

        # print(f'Initial context: "{prompt_part_1}"')
        # print(f"Number of context words (tokens): {n_ctx_1}")

        self.ctx_1 = nn.Parameter(ctx_vectors_1)  # to be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        self.tokenized_prompts = {}
        embedding = {}

        self.token_prefix = {}
        self.token_suffix = {}

        for class_name in classnames:

            p = [
                prompt_part_1
                + " "
                + status_i.format(class_name)
                + "."
                for status_i in status
            ]

            
            with torch.no_grad():
                self.tokenized_prompts[class_name] = tokenizer(p).to(device)
                embedding[class_name] = clip_model.token_embedding(
                    self.tokenized_prompts[class_name]
                )

            self.token_prefix[class_name] = embedding[class_name][:, :1, :]
            self.token_prefix[class_name].requires_grad = False


            self.token_suffix[class_name] = embedding[class_name][
                :, 1 + n_ctx_1:, :
            ]
            self.token_suffix[class_name].requires_grad = False

    def construct_prompts(self, ctx_1, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx_1,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, class_name):

        ctx_1 = self.ctx_1
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx_1 = ctx_1.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted_1 = ctx_1 + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]
        suffix = self.token_suffix[class_name]

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_1i in ctx_shifted_1:
            ctx_1i = ctx_shifted_1i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_1i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)

        return prompts
    


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding  # .type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # .type(self.dtype)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x
    
class Adapter(nn.Module):
    def __init__(self, c_in, reduction = 2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.SiLU()
        )

    def forward(self, x):
        y = self.fc(x) 
        return y

class LinearLayer_fc(nn.Module):
    def __init__(self, dim_in, dim_out, k, clip_model):
        super(LinearLayer_fc, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                assert 0 == 1
        return tokens



# class Dinov2Featurizer(nn.Module):

#     def __init__(self,):
#         super().__init__()
#         self.dim = 384 #dim
#         patch_size = 14 #self.cfg.dino_patch_size
#         self.patch_size = patch_size
#         self.feat_type = 'feat'#self.cfg.dino_feat_type
#         arch = 'vit_small' #self.cfg.model_type
#         self.model = vitsv2.__dict__[arch](
#             patch_size=patch_size,)
#         for p in self.model.parameters():
#             p.requires_grad = False
#         self.model.eval().cuda()
#         self.dropout = torch.nn.Dropout2d(p=.1)
#         self.whetherdropout = False
#         if arch == "vit_small" and patch_size == 14:
#             url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
#         elif arch == "vit_base" and patch_size == 14:
#             url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
#         elif arch == "vit_large" and patch_size == 14:
#             url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
#         else:
#             raise ValueError("Unknown arch and patch size")

#         print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
#         state_dict = torch.hub.load_state_dict_from_url(url=url)
#         self.model.load_state_dict(state_dict, strict=True)

#         if arch == "vit_small":
#             self.n_feats = 384
#         else:
#             self.n_feats = 768

#     def forward(self, img, n=1, return_class_feat=False):
#         self.model.eval()
#         with torch.no_grad():
#             assert (img.shape[2] % self.patch_size == 0)
#             assert (img.shape[3] % self.patch_size == 0)

#             # get selected layer activations
#             feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
#             feat, attn, qkv = feat[0], attn[0], qkv[0]

#             feat_h = img.shape[2] // self.patch_size
#             feat_w = img.shape[3] // self.patch_size

#             if self.feat_type == "feat":
#                 image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
#             elif self.feat_type == "KK":
#                 image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
#                 B, H, I, J, D = image_k.shape
#                 image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
#             else:
#                 raise ValueError("Unknown feat type:{}".format(self.feat_type))

#             if return_class_feat:
#                 return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

#         code = image_feat

#         if self.whetherdropout:
#             return self.dropout(image_feat), code
#         else:
#             return image_feat, code