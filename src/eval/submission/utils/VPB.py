import torch 
from torch import nn
import numpy as np
from torch.nn import functional as F 

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut"
}
class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            #nn.Linear(input_units, output_units, bias=True),
            nn.Linear(input_units, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out



class Fuse_Block(nn.Module):
    def __init__(self, dim_i,  dim_hid, dim_out):
        super(Fuse_Block, self).__init__()

        self.pre_process = nn.Sequential(nn.Linear(dim_i, dim_hid), nn.ReLU(), nn.Linear(dim_hid, dim_hid))
        self.post_process = nn.Linear(dim_hid, dim_out)
    
    def forward(self, x):
        x = self.pre_process(x)
        x = torch.mean(x, dim = 1)
        x = self.post_process(x)
        return x
    

class Zero_Parameter(nn.Module):

    def __init__(self, dim_v,dim_t, dim_out, num_heads = 4, k = 4):
        super().__init__()
        self.num_heads = num_heads 
        self.head_dim = dim_out // num_heads 
        self.dim_out = dim_out
        self.scale = dim_out ** -0.5        
        self.linear_proj_q = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.linear_proj_k = nn.Conv1d(dim_v, dim_t, kernel_size=1)
        self.linear_proj_v = nn.Conv1d(dim_v, dim_t, kernel_size=1)

        self.beta_t = 1    #1
        self.beta_s = 1    #2
        
    def forward(self, F_t, F_s):

        F_s = torch.cat(F_s, dim = 1)
        B1, N1, C1 = F_t.shape
        B2, N2, C2 = F_s.shape
        assert B1 == B2


        q_t = self.linear_proj_q(F_t.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)
        k_s = self.linear_proj_k(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)
        v_s = self.linear_proj_v(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)
        attn_t = torch.einsum('bnkc,bmkc->bknm', q_t, k_s) * self.scale
        attn_t = attn_t.softmax(dim = -1)
        F_t_a = torch.einsum('bknm,bmkc->bnkc', attn_t, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = F_t_a + F_t
        return F_t_a




class Global_Feature(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out, k):

        super(Global_Feature, self).__init__()
        
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.compress =  nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)


    def forward(self, inps):
        x = torch.cat(inps, dim = 2)
        x = self.fuse_modules(x)
        x_temp = self.compress(x)
        attention_weights = nn.Softmax(dim=1)(x_temp) 
        x = torch.sum(attention_weights * x, dim=1)
        x = self.post_process(x)
        return x





class Context_Prompting(nn.Module):
    def __init__(self):
        super().__init__()

        self.prompt_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prompt_temp_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vision_width = 1024
        self.text_width = 768
        self.fuse = Global_Feature(self.vision_width, self.vision_width // 2, self.text_width, k = 4)
        self.RCA = Zero_Parameter(dim_v = self.vision_width, dim_t = self.text_width, dim_out= self.text_width, k = 4)
        self.class_mapping = nn.Linear(self.text_width, self.text_width)
        self.image_mapping = nn.Linear(self.text_width, self.text_width)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # 使用默认的初始化方法来初始化可训练的权重
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # 避免死亡 ReLU 问题，使用小的偏置值来初始化
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                # 初始化层归一化的权重为 1，偏置为 0
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    

    
    
    # 方案1
    def forward(self, text_features, image_features, patch_tokens):
        text_features = text_features.permute(0,2,1)
        text_embeddings_mapping = self.class_mapping(text_features)
        text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
        image_embeddings_mapping = self.image_mapping(image_features + self.fuse(patch_tokens))
        image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
        pro_img = self.prompt_temp_1.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)
        return pro_img
    
    

    '''
    # 方案2
    def forward(self, text_features, image_features, patch_tokens):
        text_features = text_features.permute(0,2,1)
        text_features_new = self.RCA(text_features, patch_tokens)
        text_embeddings_mapping = self.class_mapping(text_features_new)
        text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
        image_embeddings_mapping = self.image_mapping(image_features + self.fuse(patch_tokens))
        image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
        pro_img = self.prompt_temp_1.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)
        return pro_img
    '''




