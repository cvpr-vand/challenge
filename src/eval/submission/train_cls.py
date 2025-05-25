import argparse

import torch.optim.lr_scheduler

# from .datasets import *
# from .datasets import dataset_classes
from .utils.csv_utils import *
from .utils.metrics import *
from .utils.training_utils import *
from .PromptAD import *
from .utils.eval_utils import *
from torchvision import transforms
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

TASK = 'CLS'

dataset_classes = ['breakfast_box', 'screw_bag', 'pushpins', 
               'splicing_connectors', 'juice_bottle']


def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)

# def mmd_loss(f1, f2, kernel_mul=2.0, kernel_num=5):
#         batch_size = min(f1.size(0), f2.size(0))
#         kernels = []
#         for i in range(kernel_num):
#             bandwidth = kernel_mul ** i
#             kernels.append(GaussianKernel(bandwidth))
#         loss = 0
#         for kernel in kernels:
#             k1 = kernel(f1[:batch_size], f1[:batch_size])
#             k2 = kernel(f2[:batch_size], f2[:batch_size])
#             k12 = kernel(f1[:batch_size], f2[:batch_size])
#             loss += (k1.mean() + k2.mean() - 2*k12.mean())
#         return loss / kernel_num

def validate_gradient_flow(model, device="cuda"):
    # 准备模型
    model = model.to(device)
    model.train()  # 确保训练模式
    
    # 打印可训练参数
    print("="*50 + "\n可训练参数清单:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} | 形状: {param.shape}")
    
    # 生成测试数据
    inputs = torch.randn(2, 3, 240, 240).to(device)
    targets = torch.randn(2, 240, 240).to(device)
    
    # 前向传播
    outputs, _ = model(inputs,'cls')
    # outputs = torch.tensor(outputs, device=device)
    
    loss = 0 
    loss = [(loss + f * 0.9) for f in outputs]
    loss = torch.stack(loss, dim=0).to(device)
    
    # 反向传播前梯度状态
    print("\n" + "="*50 + "\n反向传播前梯度:")
    for name, param in model.named_parameters():
        print(f"{name} - 梯度存在: {param.grad is not None}")
    
    # 执行反向传播
    loss.backward()
    
    # 反向传播后梯度状态
    print("\n" + "="*50 + "\n反向传播后梯度:")
    grad_status = []
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        grad_status.append(has_grad)
        grad_norm = param.grad.norm().item() if has_grad else 0
        print(f"{name[:30]}... | 梯度存在: {has_grad} | 梯度范数: {grad_norm:.4f}")
    
    # 参数更新验证
    initial_params = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    
    print("\n" + "="*50 + "\n参数更新验证:")
    for name, param in model.named_parameters():
        delta = torch.abs(param.data - initial_params[name]).mean()
        print(f"{name[:30]}... | 参数变化均值: {delta:.6f}")
    
    return any(grad_status)  # 返回是否存在有效梯度

def fit(model,
        args,
        # dataloader: DataLoader,
        device: str,
        check_path: str,
        # train_data: DataLoader,
        train_data: torch
        ):

    # change the model into eval mode
    model.eval_mode()

    # has_grad = validate_gradient_flow(model)
    # print(f"\n梯度传播验证结果: {'成功' if has_grad else '失败'}")

    # input()

    features1 = []
    features2 = []
    # for (data, mask, label, name, img_type) in train_data:

    #     data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
    #     data = torch.stack(data, dim=0).to(device)
    #     _, _, feature_map1, feature_map2 = model.encode_image(data)
    #     features1.append(feature_map1)
    #     features2.append(feature_map2)

    ######
    data = train_data
    # print("type of data: ", data.type)
    # for f in data:
        # print("tensor.shape: ", f.shape)
    # data = [model.transform(Image.fromarray(cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_BGR2RGB))) for f in data]
    # data = [model.transform(Image.fromarray(cv2.cvtColor((f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))) for f in data]
    data = [
        model.transform(
            Image.fromarray(
                (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))) 
                for f in data]
    # for f in data:
    #     print("tensor.shape after transfer: ", f.shape)
    data = torch.stack(data, dim=0).to(device)
    # print("shape of data: ", data.shape)
    # data = torch.to(device)
    _, _, feature_map1, feature_map2 = model.encode_image(data)
    features1.append(feature_map1)
    features2.append(feature_map2)
    ######

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

    # print()

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['Epoch'], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    for epoch in range(args['Epoch']):
        ######
    # for (data, mask, label, name, img_type) in train_data:
        data = train_data
        # data = [model.transform(Image.fromarray(cv2.cvtColor(f.cpu().numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = [
            model.transform(
                Image.fromarray(
                    (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))) 
                    for f in data]
        data = torch.stack(data, dim=0).to(device)
        # print("shape of data: ", data.shape)
        # data = data.to(device)

        data = data[0:1, :, :, :].to(device)
        # data = data.to(device)

        normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

        optimizer.zero_grad()

        normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

        abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
        abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
        abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

        # compute mean
        mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
        mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

        loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

        cls_feature, _, _, _ = model.encode_image(data)

        # compute v2t loss and triplet loss
        normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
        normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

        abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
        abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

        l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
        l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

        if model.precision == 'fp16':
            logit_scale = model.model.logit_scale.half()
        else:
            logit_scale = model.model.logit_scalef

        logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

        target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

        loss_v2t = criterion(logits_v2t, target_v2t)

        trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
        loss = loss_v2t + trip_loss + loss_match_abnormal * args['lambda1']
        print("loss: ", loss)
        loss.backward()

        # grad_status = []
        # for name, param in model.named_parameters():
        #     has_grad = param.grad is not None
        #     grad_status.append(has_grad)
        #     grad_norm = param.grad.norm().item() if has_grad else 0
        #     print(f"{name[:30]}... | 梯度存在: {has_grad} | 梯度范数: {grad_norm:.4f}")

        optimizer.step()

        scheduler.step()
        model.build_text_feature_gallery()

        _ , _ , check_path= get_dir_from_args('CLS', **args)

        save_check_point(model, check_path)


    # return best_result_dict

    # for epoch in range(args['Epoch']):
    #     # 数据预处理部分保持不变
    #     data = train_data
    #     data = [
    #         model.transform(
    #             Image.fromarray(
    #                 (f.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))) 
    #                 for f in data]
    #     data = torch.stack(data, dim=0).to(device)
    #     data = data[0:1, :, :, :].to(device)  # 实际使用时建议移除切片操作
        
    #     # 文本提示生成
    #     normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()
        
    #     # 优化器初始化
    #     optimizer.zero_grad()
        
    #     # 文本特征编码
    #     normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)
    #     abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
    #     abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
    #     abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
        
    #     # ========== 修改1：使用MMD替代原对齐损失 ==========

    #     loss_match_abnormal = mmd_loss(
    #         F.normalize(abnormal_text_features_handle, dim=-1),
    #         F.normalize(abnormal_text_features_learned, dim=-1)
    #     )
        
    #     # 图像特征编码
    #     cls_feature, _, _, _ = model.encode_image(data)
        
    #     # ========== 修改2：对称对比损失计算 ==========
    #     # 文本和图像特征归一化
    #     normal_text_anchor = F.normalize(normal_text_features.mean(dim=0, keepdim=True), dim=-1)
    #     abnormal_text_anchor = F.normalize(abnormal_text_features.mean(dim=0, keepdim=True), dim=-1)
    #     abnormal_text_all = F.normalize(abnormal_text_features, dim=-1)
    #     image_features = F.normalize(cls_feature, dim=-1)
        
    #     # 图像到文本对比 (V2T)
    #     logits_v2t = torch.matmul(image_features, torch.cat([normal_text_anchor, abnormal_text_all], 0).T)
    #     logits_v2t *= model.model.logit_scale.exp()
        
    #     # 文本到图像对比 (T2V)
    #     logits_t2v = torch.matmul(normal_text_anchor, torch.cat([image_features, image_features], 1).T)
    #     logits_t2v *= model.model.logit_scale.exp()
        
    #     # 对比损失计算
    #     labels = torch.zeros(logits_v2t.size(0), dtype=torch.long).to(device)
    #     loss_v2t = F.cross_entropy(logits_v2t, labels)
    #     loss_t2v = F.cross_entropy(logits_t2v, labels)
    #     loss_contrastive = (loss_v2t + loss_t2v) * 0.5
        
    #     # ========== 修改3：使用Circle Loss替代三元组损失 ==========
    #     pos_sim = torch.sum(image_features * normal_text_anchor, dim=1)  # [B]
    #     neg_sim = torch.sum(image_features * abnormal_text_anchor, dim=1)  # [B]
        
    #     margin = 0.25
    #     gamma = 64  # 缩放因子
    #     loss_circle = torch.log(1 + torch.exp(gamma * (neg_sim - pos_sim + margin)))
    #     loss_circle = loss_circle.mean()
        
    #     # ========== 总损失计算 ==========
    #     total_loss = (
    #         loss_contrastive +
    #         loss_circle  +
    #         loss_match_abnormal * 0.5
    #     )

    #     print("total loss: ", total_loss)
        
    #     # 反向传播
    #     total_loss.backward()
    #     optimizer.step()

    #     scheduler.step()
    #     model.build_text_feature_gallery()

    #     save_check_point(model, check_path)

    # return best_result_dict


def main(args):
    # kwargs = vars(args)
    kwargs = args

    # if kwargs['seed'] is None:
        # kwargs['seed'] = 111

    # setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    # train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # print("train_dataloader.shape: ", train_dataloader.shape)
    # 测试data是否正常
    # for data in train_dataloader:
    #     print("Input image shape:", data[0].shape)

    # get the test dataloader
    print("category: ", kwargs['class_name'])
    print("dataset classes names: ", dataset_classes)
    # test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    train_data = kwargs['train_data']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    # metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)
    # metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_data)

    # metrics = fit(model, args, device, check_path=check_path, train_data=train_data)
    fit(model, args, device, check_path=check_path, train_data=train_data)

    # i_roc = round(metrics['i_roc'], 2)
    # object = kwargs['class_name']
    # print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    # save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
    #             kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# def get_args():
#     parser = argparse.ArgumentParser(description='Anomaly detection')
#     parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'mvtecloco'])
#     parser.add_argument('--class_name', type=str, default='carpet')

#     # ViT-B-16-plus-240
#     parser.add_argument('--img-resize', type=int, default=240)
#     parser.add_argument('--img-cropsize', type=int, default=240)
#     # ViT-L-14-336
#     # parser.add_argument('--img-resize', type=int, default=336)
#     # parser.add_argument('--img-cropsize', type=int, default=336)
#     parser.add_argument('--resolution', type=int, default=400)

#     parser.add_argument('--batch-size', type=int, default=400)
#     parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
#     parser.add_argument("--root-dir", type=str, default="./result")
#     parser.add_argument("--load-memory", type=str2bool, default=True)
#     parser.add_argument("--cal-pro", type=str2bool, default=False)
#     parser.add_argument("--seed", type=int, default=111)
#     parser.add_argument("--gpu-id", type=int, default=0)

#     # pure test
#     parser.add_argument("--pure-test", type=str2bool, default=False)

#     # method related parameters
#     parser.add_argument('--k-shot', type=int, default=1)
#     parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
#                         choices=['ViT-B-16-plus-240', 'ViT-B-16', "ViT-B-32", "ViT-L-14", "ViT-L-14-336"])
#                         # choices=['ViT-B-16-plus-240', 'ViT-B-16'])
#     parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
#     # parser.add_argument("--pretrained_dataset", type=str, default="openai")

#     parser.add_argument("--use-cpu", type=int, default=0)

#     # prompt tuning hyper-parameter
#     parser.add_argument("--n_ctx", type=int, default=4)
#     parser.add_argument("--n_ctx_ab", type=int, default=1)
#     parser.add_argument("--n_pro", type=int, default=3)
#     parser.add_argument("--n_pro_ab", type=int, default=4)
#     parser.add_argument("--Epoch", type=int, default=500)
#     # optimizer
#     parser.add_argument("--lr", type=float, default=0.002)
#     parser.add_argument("--momentum", type=float, default=0.9)
#     parser.add_argument("--weight_decay", type=float, default=0.0005)

#     # loss hyper parameter
#     parser.add_argument("--lambda1", type=float, default=0.001)

#     args = parser.parse_args()

#     return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    main(args)
