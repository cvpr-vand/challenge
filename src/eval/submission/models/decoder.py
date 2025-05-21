import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, clip_model):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_out, dim_out) for _ in range(k)])
        self.proj = clip_model.visual.proj

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:

                tokens[i] = tokens[i][:, 1:, :] @ self.proj
                tokens[i] = self.fc[i](tokens[i]) #+ tokens[i]
            else:
                assert 0 == 1

        return tokens

    
class LinearLayer_no_fc(nn.Module):
    def __init__(self, clip_model):
        super(LinearLayer_no_fc, self).__init__()
        self.proj = clip_model.visual.proj

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i][:, 1:, :] @ self.proj
            else:
                assert 0 == 1

        return tokens




class CovLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer, self).__init__()
        self.fc_33 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")
                for i in range(k)
            ]
        )
        self.fc_11 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")
                for i in range(k)
            ]
        )
        self.fc_77 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")
                for i in range(k)
            ]
        )
        self.fc_51 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same")
                for i in range(k)
            ]
        )
        self.fc_15 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same")
                for i in range(k)
            ]
        )

        self.fc_55 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")
                for i in range(k)
            ]
        )


    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                x = tokens[i][:, 1:, :]
                x = x.view(
                    x.shape[0],
                    int(np.sqrt(x.shape[1])),
                    int(np.sqrt(x.shape[1])),
                    x.shape[2],
                )
                # print(x.shape)
                x_temp = (
                    self.fc_11[i](x.permute(0, 3, 1, 2))
                    + self.fc_33[i](x.permute(0, 3, 1, 2))
                    + self.fc_55[i](x.permute(0, 3, 1, 2))
                    + self.fc_77[i](x.permute(0, 3, 1, 2))
                    + self.fc_15[i](x.permute(0, 3, 1, 2))
                    + self.fc_51[i](x.permute(0, 3, 1, 2))
                )
                tokens[i] = x_temp
                tokens[i] = (
                    tokens[i]
                    .permute(0, 2, 3, 1)
                    .view(tokens[i].shape[0], -1, tokens[i].shape[1])
                )
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](
                    tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()
                )
        return tokens
    

# class Adapter(nn.Module):
#     def __init__(self, c_in, reduction = 2):
#         super(Adapter, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, c_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // reduction, c_in, bias=False),
#             nn.SiLU()
#         )

#     def forward(self, x):
#         y = self.fc(x)
#         x = x + y 
#         return x
    
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
    
class Adapter_linear(nn.Module):
    def __init__(self, c_in, reduction = 2):
        super(Adapter_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in),
        )

    def forward(self, x):
        y = self.fc(x) 
        return y
    
class Adapter_linear_residual(nn.Module):
    def __init__(self, c_in, reduction = 2):
        super(Adapter_linear_residual, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in),
        )

    def forward(self, x):
        y = self.fc(x) + x
        return y
    
class Adapter_residual(nn.Module):
    def __init__(self, c_in, reduction = 2):
        super(Adapter_residual, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.SiLU()
        )

    def forward(self, x):
        y = self.fc(x) + x
        return y


class LinearLayer_no_proj(nn.Module):
    def __init__(self, dim_in, dim_out, k, clip_model):
        super(LinearLayer_no_proj, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
        self.proj = clip_model.visual.proj

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                # tokens[i] =  @ self.proj
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                assert 0 == 1

        return tokens
    
class CovLayer_only_scale(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_only_scale, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                + self.fc_55[i](x.permute(0, 3, 1, 2))
                + self.fc_77[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens
    

class CovLayer_only_shape(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer, self).__init__()
        self.fc_51 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same")
                for i in range(k)
            ]
        )
        self.fc_15 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same")
                for i in range(k)
            ]
        )


    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                x = tokens[i][:, 1:, :]
                x = x.view(
                    x.shape[0],
                    int(np.sqrt(x.shape[1])),
                    int(np.sqrt(x.shape[1])),
                    x.shape[2],
                )
                x_temp = (
                    + self.fc_15[i](x.permute(0, 3, 1, 2))
                    + self.fc_51[i](x.permute(0, 3, 1, 2))
                )
                tokens[i] = x_temp
                tokens[i] = (
                    tokens[i]
                    .permute(0, 2, 3, 1)
                    .view(tokens[i].shape[0], -1, tokens[i].shape[1])
                )
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](
                    tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()
                )
        return tokens
    

class CovLayer_11_33(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_11_33, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        # self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        # self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                # + self.fc_55[i](x.permute(0, 3, 1, 2))
                # + self.fc_77[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens
    

class CovLayer_11_33_55(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_11_33_55, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        # self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                + self.fc_55[i](x.permute(0, 3, 1, 2))
                # + self.fc_77[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens
    

class CovLayer_11_33_55_77(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_11_33_55_77, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                + self.fc_55[i](x.permute(0, 3, 1, 2))
                + self.fc_77[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens
    
class CovLayer_11_33_55_77_99(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_11_33_55_77_99, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])
        self.fc_99 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=9, padding="same")for i in range(k)])


    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                + self.fc_55[i](x.permute(0, 3, 1, 2))
                + self.fc_77[i](x.permute(0, 3, 1, 2))
                + self.fc_99[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens
    
class CovLayer_33_55(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer_33_55, self).__init__()
        self.fc_33 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")for i in range(k)])
        # self.fc_11 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")for i in range(k)])
        # self.fc_77 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")for i in range(k)])
        self.fc_55 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")for i in range(k)])
        # self.fc_99 = nn.ModuleList([nn.Conv2d(dim_in, dim_out, kernel_size=9, padding="same")for i in range(k)])


    def forward(self, tokens):
        for i in range(len(tokens)):
            x = tokens[i][:, 1:, :]
            x = x.view(
                x.shape[0],
                int(np.sqrt(x.shape[1])),
                int(np.sqrt(x.shape[1])),
                x.shape[2],
            )
            x_temp = (
                # self.fc_11[i](x.permute(0, 3, 1, 2))
                + self.fc_33[i](x.permute(0, 3, 1, 2))
                + self.fc_55[i](x.permute(0, 3, 1, 2))
                # + self.fc_77[i](x.permute(0, 3, 1, 2))
                # + self.fc_99[i](x.permute(0, 3, 1, 2))
            )
            tokens[i] = x_temp
            tokens[i] = (
                tokens[i]
                .permute(0, 2, 3, 1)
                .view(tokens[i].shape[0], -1, tokens[i].shape[1])
            )
        return tokens