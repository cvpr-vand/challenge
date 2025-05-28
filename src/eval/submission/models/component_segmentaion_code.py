import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import morphology, measure
import tqdm

import torch
import yaml
from scipy import ndimage
from PIL import Image, ImageEnhance

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def turn_binary_to_int(mask):
    temp = np.where(mask,255,0).astype(np.uint8)
    return temp

def crop_by_mask(image,mask):
        if image.shape[-1] == 3:
            mask = cv2.merge([mask,mask,mask])
        return np.where(mask!=0,image,0)

def remove_background(background,masks):
    result = list()
    for mask in masks:
        # if np.sum(np.logical_and(mask,background)) == 0:
        # cv2.imshow(f"mask{intersect_ratio(mask,background)}",mask)
        # cv2.waitKey(0)
        if intersect_ratio(mask,background) < 0.95:
            result.append(mask)
    return result

def save_tensor_as_jpg(tensor_images, save_dir="saved_images", class_name=None):

    os.makedirs(save_dir, exist_ok=True)
    

    for i in range(tensor_images.shape[0]):

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


            x2 = min(x2, data.shape[1])
            y2 = min(y2, data.shape[0])


            black_image[y1:y2, x1:x2] = data[y1:y2, x1:x2]
            data = black_image

            alpha = 1.5  
            beta = 10    
            contrast_enhanced = cv2.convertScaleAbs(data, alpha=alpha, beta=beta)

            kernel = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)

            cv2.imwrite(f"{save_dir}/sharpened_{i}.jpg", sharpened)

        if class_name == "screw_bag":
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            blurred_bg = cv2.GaussianBlur(clahe_img, (101, 101), 0)
            normalized_img = cv2.addWeighted(clahe_img, 1.5, blurred_bg, -1.5, 20)
            clahe_bgr = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
            pil_img = Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
            enhancer_sharp = ImageEnhance.Sharpness(pil_img)
            sharp_img = enhancer_sharp.enhance(2.5)
            sharpened = cv2.cvtColor(np.array(sharp_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/sharpened_{i}.jpg", sharpened)

        cv2.imwrite(f"{save_dir}/image_{i}.jpg", data)

def merge_small_regions_to_smallest_neighbor(mask_list, min_area_threshold=2000, max_target_area_threshold=6000, merge_to_kth_largest=-1):
    """
    将面积小于阈值的完整mask合并到相邻的倒数第k大的mask中，直到所有mask都超过阈值
    
    参数:
    mask_list: 长度为k的列表，每个元素是[256, 256]的numpy数组，值为0或255
    min_area_threshold: 最小面积阈值，小于此面积的整个mask将被合并
    max_target_area_threshold: 最大目标面积阈值，面积大于此阈值的mask将不会作为合并目标
                              设为None表示不限制
    merge_to_kth_largest: 合并到相邻mask中的倒数第k大者（1表示最小，2表示第二小，以此类推）
                         如果设为-1，则合并到面积最大的邻居
    min_neighbor_area: 允许合并的邻居mask的最小面积。设为0则不限制
    
    返回:
    处理后的mask列表，长度可能小于k（因为一些mask可能被合并）
    """
    # 复制输入的masks，避免修改原始数据
    current_masks = [mask.copy() for mask in mask_list]
    
    # 初始化迭代计数器，防止无限循环
    iteration = 0
    max_iterations = 5  # 设置最大迭代次数
    
    # 获取mask尺寸
    height, width = current_masks[0].shape
    
    # 计算初始面积，用于确定哪些mask不应被合并到
    initial_mask_areas = [np.sum(mask > 0) for mask in current_masks]
    
    # 确定哪些mask不能作为合并目标（即面积大于max_target_area_threshold的mask）
    excluded_targets = set()
    if max_target_area_threshold is not None:
        excluded_targets = {i for i, area in enumerate(initial_mask_areas) if area > max_target_area_threshold}
        # print(f"排除 {len(excluded_targets)} 个面积大于 {max_target_area_threshold} 的mask作为合并目标")
    
    while iteration < max_iterations:
        # 计算每个mask的面积
        mask_areas = [np.sum(mask > 0) for mask in current_masks]
        # print(mask_areas)
        # 找出小于阈值的mask索引
        small_mask_indices = [i for i, area in enumerate(mask_areas) if area < min_area_threshold]
        
        # 如果没有小于阈值的mask，结束循环
        if len(small_mask_indices) == 0:
            break
        
        # 将所有mask组合成一个带标签的图像
        # 每个mask的区域标记为mask的索引+1（0保留为背景）
        volume = np.zeros((height, width), dtype=np.int32)
        for i, mask in enumerate(current_masks):
            # 只在当前没有标签的位置添加新标签
            mask_region = (mask > 0)
            volume = np.where(volume == 0, (i + 1) * mask_region, volume)
        
        # 标记已处理的mask
        processed_masks = set()
        any_merges = False  # 标记是否进行了任何合并
        
        # 获取小于阈值的mask的面积和索引对
        small_areas_with_indices = [(mask_areas[i], i) for i in small_mask_indices]
        
        # 根据面积从小到大排序小mask索引
        sorted_small_indices = [idx for _, idx in sorted(small_areas_with_indices, reverse=True)]
        
        # 处理每个小mask
        for idx in sorted_small_indices:
            # 如果这个mask已经被处理过，跳过
            if idx in processed_masks:
                continue
            
            # 获取当前小mask
            small_mask = (volume == (idx + 1))
            
            # 如果这个mask已经被之前的操作合并了，跳过
            if not np.any(small_mask):
                processed_masks.add(idx)
                continue
            
            area_now = np.sum(small_mask)
            # 扩张小mask以找到邻居
            dilated = morphology.binary_dilation(small_mask)
            neighbor_mask = dilated & ~small_mask
            
            # 找到邻居mask的标签
            neighbor_labels = np.unique(volume[neighbor_mask])
            
            # 过滤掉背景(0)和自身
            neighbor_labels = [l for l in neighbor_labels if l != 0 and l != (idx + 1)]
            
            # 过滤掉面积小于min_neighbor_area的邻居
            # if min_neighbor_area > 0:
            #     neighbor_labels = [l for l in neighbor_labels if mask_areas[l-1] >= area_now]
            
            # 过滤掉那些被排除的大mask作为合并目标
            neighbor_labels = [l for l in neighbor_labels if (l-1) not in excluded_targets]
            
            if neighbor_labels:
                # 找到倒数第k大的邻居
                if merge_to_kth_largest > 0:
                    # 按面积排序邻居
                    neighbor_areas = [(l, mask_areas[l-1]) for l in neighbor_labels]
                    neighbor_areas.sort(key=lambda x: x[1])  # 按面积从小到大排序
                    
                    # 选择倒数第k大的邻居（k=1是最小的）
                    k_idx = min(merge_to_kth_largest - 1, len(neighbor_areas) - 1)
                    target_label = neighbor_areas[k_idx][0]
                elif merge_to_kth_largest == -1:
                    # 找到面积最大的邻居
                    target_label = max(neighbor_labels, key=lambda l: mask_areas[l-1])
                else:
                    # 参数无效，默认使用最小的邻居
                    # print("警告: merge_to_kth_largest参数无效，使用最小的邻居")
                    target_label = min(neighbor_labels, key=lambda l: mask_areas[l-1])
                
                # 将小mask合并到目标mask
                volume[small_mask] = target_label
                
                # 更新mask面积
                mask_areas[target_label-1] += mask_areas[idx]
                mask_areas[idx] = 0
                
                processed_masks.add(idx)
                any_merges = True
            else:
                pass

            # break
                # 所有邻居都不满足条件
                # print(f"警告: mask {idx} (面积: {mask_areas[idx]}) 没有满足条件的邻居可以合并。")
        
        # 如果这次迭代没有进行任何合并，但仍有小于阈值的mask，
        # 说明这些小mask无法合并（可能是孤立的或没有满足条件的邻居），我们应该停止循环
        # if not any_merges and len(small_mask_indices) > 0:
        #     print(f"警告: 有 {len([i for i in small_mask_indices if i not in processed_masks])} 个小mask无法合并（孤立或无符合条件的邻居）。")
        #     break
        
        # 从volume重建masks
        unique_labels = np.unique(volume)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景
        
        new_masks = []
        for label in unique_labels:
            new_mask = (volume == label).astype(np.uint8) * 255
            new_masks.append(new_mask)
        
        # 更新当前masks
        current_masks = new_masks
        
        iteration += 1
        # print(f"迭代 {iteration}: 剩余 {len(current_masks)} 个mask")
    
    # if iteration == max_iterations:
        # print(f"警告: 达到最大迭代次数 ({max_iterations})，可能仍有小于阈值的mask。")
    
    return current_masks
    


def smooth_masks(mask_list, closing_kernel_size=5, opening_kernel_size=3):
    """
    平滑mask边缘，移除噪点和毛刺
    
    参数:
    mask_list: 长度为k的列表，每个元素是[H, W]的numpy数组，值为0或255
    closing_kernel_size: 闭运算的核大小，用于填充小孔洞
    opening_kernel_size: 开运算的核大小，用于移除小毛刺
    
    返回:
    处理后的mask列表
    """
    smoothed_masks = []
    
    for i, mask in enumerate(mask_list):
        # 先闭运算（先膨胀后腐蚀），填充小孔
        closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # 再开运算（先腐蚀后膨胀），移除小毛刺
        opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_kernel)
        
        # 确保二值化
        _, smooth_mask = cv2.threshold(opened_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 可选: 对于非常不规则的边缘，可以使用高斯模糊后再二值化
        # blur_mask = cv2.GaussianBlur(smooth_mask, (5, 5), 0)
        # _, smooth_mask = cv2.threshold(blur_mask, 127, 255, cv2.THRESH_BINARY)
        
        smoothed_masks.append(smooth_mask)
        
    return smoothed_masks

def advanced_smooth_masks(mask_list, contour_smoothing=True, remove_small_holes=True, 
                        remove_small_objects=True, hole_size=50, object_size=50):
    """
    高级平滑处理，包括轮廓平滑、移除小孔洞和小物体
    
    参数:
    mask_list: 长度为k的列表，每个元素是[H, W]的numpy数组，值为0或255
    contour_smoothing: 是否平滑轮廓
    remove_small_holes: 是否移除小孔洞
    remove_small_objects: 是否移除小物体
    hole_size: 小孔洞的最大面积
    object_size: 小物体的最大面积
    
    返回:
    处理后的mask列表
    """
    processed_masks = []
    
    for i, mask in enumerate(mask_list):
        # 转换为二值图像
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 轮廓平滑处理
        if contour_smoothing:
            # 找到所有轮廓
            contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建一个空白图像
            smooth_contour_mask = np.zeros_like(binary_mask)
            
            # 对每个轮廓应用多边形近似
            for contour in contours:
                epsilon = 0.001 * cv2.arcLength(contour, True)  # 使用周长的0.1%作为精度
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # 为了更平滑的效果，可以用B样条曲线
                # 但OpenCV没有直接的B样条实现，所以我们用填充轮廓的方式
                cv2.drawContours(smooth_contour_mask, [smoothed_contour], 0, 1, -1)
            
            binary_mask = smooth_contour_mask
        
        # 移除小孔洞
        if remove_small_holes:
            binary_mask = morphology.remove_small_holes(binary_mask.astype(bool), area_threshold=hole_size).astype(np.uint8)
        
        # 移除小物体
        if remove_small_objects:
            binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=object_size).astype(np.uint8)
        
        # 转换回原始格式
        processed_mask = binary_mask * 255
        processed_masks.append(processed_mask)
    
    return processed_masks


def merge_small_regions_to_smallest_neighbor_one_time(mask_list, min_area_threshold=2000, max_target_area_threshold=6000, merge_to_kth_largest=-1):
    """
    将面积小于阈值的完整mask合并到相邻的倒数第k大的mask中，直到所有mask都超过阈值
    
    参数:
    mask_list: 长度为k的列表，每个元素是[256, 256]的numpy数组，值为0或255
    min_area_threshold: 最小面积阈值，小于此面积的整个mask将被合并
    max_target_area_threshold: 最大目标面积阈值，面积大于此阈值的mask将不会作为合并目标
                              设为None表示不限制
    merge_to_kth_largest: 合并到相邻mask中的倒数第k大者（1表示最小，2表示第二小，以此类推）
                         如果设为-1，则合并到面积最大的邻居
    min_neighbor_area: 允许合并的邻居mask的最小面积。设为0则不限制
    
    返回:
    处理后的mask列表，长度可能小于k（因为一些mask可能被合并）
    """
    # 复制输入的masks，避免修改原始数据
    current_masks = [mask.copy() for mask in mask_list]
    
    # 初始化迭代计数器，防止无限循环
    iteration = 0
    max_iterations = 10  # 设置最大迭代次数
    
    # 获取mask尺寸
    height, width = current_masks[0].shape
    
    # 计算初始面积，用于确定哪些mask不应被合并到
    initial_mask_areas = [np.sum(mask > 0) for mask in current_masks]
    
    # 确定哪些mask不能作为合并目标（即面积大于max_target_area_threshold的mask）
    excluded_targets = set()
    if max_target_area_threshold is not None:
        excluded_targets = {i for i, area in enumerate(initial_mask_areas) if area > max_target_area_threshold}
        # print(f"排除 {len(excluded_targets)} 个面积大于 {max_target_area_threshold} 的mask作为合并目标")
    
    while iteration < max_iterations:
        # 计算每个mask的面积
        mask_areas = [np.sum(mask > 0) for mask in current_masks]
        # print(mask_areas)
        # 找出小于阈值的mask索引
        small_mask_indices = [i for i, area in enumerate(mask_areas) if area < min_area_threshold]
        
        # 如果没有小于阈值的mask，结束循环
        if len(small_mask_indices) == 0:
            break
        
        # 将所有mask组合成一个带标签的图像
        # 每个mask的区域标记为mask的索引+1（0保留为背景）
        volume = np.zeros((height, width), dtype=np.int32)
        for i, mask in enumerate(current_masks):
            # 只在当前没有标签的位置添加新标签
            mask_region = (mask > 0)
            volume = np.where(volume == 0, (i + 1) * mask_region, volume)
        
        # 标记已处理的mask
        processed_masks = set()
        any_merges = False  # 标记是否进行了任何合并
        
        # 获取小于阈值的mask的面积和索引对
        small_areas_with_indices = [(mask_areas[i], i) for i in small_mask_indices]
        
        # 根据面积从小到大排序小mask索引
        sorted_small_indices = [idx for _, idx in sorted(small_areas_with_indices, reverse=True)]
        
        # 处理每个小mask
        for idx in sorted_small_indices:
            # 如果这个mask已经被处理过，跳过
            if idx in processed_masks:
                continue
            
            # 获取当前小mask
            small_mask = (volume == (idx + 1))
            
            # 如果这个mask已经被之前的操作合并了，跳过
            if not np.any(small_mask):
                processed_masks.add(idx)
                continue
            
            area_now = np.sum(small_mask)
            # 扩张小mask以找到邻居
            dilated = morphology.binary_dilation(small_mask)
            neighbor_mask = dilated & ~small_mask
            
            # 找到邻居mask的标签
            neighbor_labels = np.unique(volume[neighbor_mask])
            
            # 过滤掉背景(0)和自身
            neighbor_labels = [l for l in neighbor_labels if l != 0 and l != (idx + 1)]
            
            # 过滤掉面积小于min_neighbor_area的邻居
            # if min_neighbor_area > 0:
            #     neighbor_labels = [l for l in neighbor_labels if mask_areas[l-1] >= area_now]
            
            # 过滤掉那些被排除的大mask作为合并目标
            neighbor_labels = [l for l in neighbor_labels if (l-1) not in excluded_targets]
            
            if neighbor_labels:
                # 找到倒数第k大的邻居
                if merge_to_kth_largest > 0:
                    # 按面积排序邻居
                    neighbor_areas = [(l, mask_areas[l-1]) for l in neighbor_labels]
                    neighbor_areas.sort(key=lambda x: x[1])  # 按面积从小到大排序
                    
                    # 选择倒数第k大的邻居（k=1是最小的）
                    k_idx = min(merge_to_kth_largest - 1, len(neighbor_areas) - 1)
                    target_label = neighbor_areas[k_idx][0]
                elif merge_to_kth_largest == -1:
                    # 找到面积最大的邻居
                    target_label = max(neighbor_labels, key=lambda l: mask_areas[l-1])
                else:
                    # 参数无效，默认使用最小的邻居
                    # print("警告: merge_to_kth_largest参数无效，使用最小的邻居")
                    target_label = min(neighbor_labels, key=lambda l: mask_areas[l-1])
                
                # 将小mask合并到目标mask
                volume[small_mask] = target_label
                
                # 更新mask面积
                mask_areas[target_label-1] += mask_areas[idx]
                mask_areas[idx] = 0
                
                processed_masks.add(idx)
                any_merges = True
            else:
                pass

            break
                # 所有邻居都不满足条件
                # print(f"警告: mask {idx} (面积: {mask_areas[idx]}) 没有满足条件的邻居可以合并。")
        
        # 如果这次迭代没有进行任何合并，但仍有小于阈值的mask，
        # 说明这些小mask无法合并（可能是孤立的或没有满足条件的邻居），我们应该停止循环
        # if not any_merges and len(small_mask_indices) > 0:
        #     print(f"警告: 有 {len([i for i in small_mask_indices if i not in processed_masks])} 个小mask无法合并（孤立或无符合条件的邻居）。")
        #     break
        
        # 从volume重建masks
        unique_labels = np.unique(volume)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景
        
        new_masks = []
        for label in unique_labels:
            new_mask = (volume == label).astype(np.uint8) * 255
            new_masks.append(new_mask)
        
        # 更新当前masks
        current_masks = new_masks
        
        iteration += 1
        # print(f"迭代 {iteration}: 剩余 {len(current_masks)} 个mask")
    
    # if iteration == max_iterations:
        # print(f"警告: 达到最大迭代次数 ({max_iterations})，可能仍有小于阈值的mask。")
    
    return current_masks


def smooth_masks_with_padding(mask_list, closing_kernel_size=20, padding=30):
    """
    添加边界进行平滑处理，然后再裁剪回原始尺寸
    """
    smoothed_masks = []
    
    for mask in mask_list:
        h, w = mask.shape
        
        # 添加边界
        padded_mask = cv2.copyMakeBorder(mask, padding, padding, padding, padding, 
                                         cv2.BORDER_CONSTANT, value=0)
        
        # 应用闭运算
        closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(padded_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # 应用开运算
        opening_kernel = np.ones((3, 3), np.uint8)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_kernel)
        
        # 裁剪回原始尺寸
        smooth_mask = opened_mask[padding:padding+h, padding:padding+w]
        
        # 确保二值化
        _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
        
        smoothed_masks.append(smooth_mask)
        
    return smoothed_masks

def post_process_masks(refined_masks, class_name):

    if class_name == "breakfast_box":
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks)
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks, min_area_threshold=2000, max_target_area_threshold=None, merge_to_kth_largest=1)
        refined_masks = smooth_masks(refined_masks)
    elif class_name == "pushpins":
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks, min_area_threshold=300, max_target_area_threshold=None, merge_to_kth_largest=1)
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks, min_area_threshold=600, max_target_area_threshold=None, merge_to_kth_largest=1)
        refined_masks = filter_masks_by_area(refined_masks, 500 * 3)
    if class_name == "juice_bottle":
        refined_masks = merge_small_regions_to_smallest_neighbor_one_time(refined_masks, min_area_threshold=200, max_target_area_threshold=None, merge_to_kth_largest=1)
    elif class_name == "screw_bag":
        refined_masks = split_masks_by_connected_component(refined_masks)
        refined_masks = filter_masks_below_area(refined_masks, 100*3)
        refined_masks = smooth_masks(refined_masks, closing_kernel_size=3)
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks, min_area_threshold=150*3, max_target_area_threshold=None, merge_to_kth_largest=-1)
        refined_masks = smooth_masks_with_padding(refined_masks, closing_kernel_size=40)
        refined_masks = merge_small_regions_to_smallest_neighbor(refined_masks, min_area_threshold=2000, max_target_area_threshold=None, merge_to_kth_largest=-1)
        refined_masks = filter_masks_below_area(refined_masks, 2000)
        refined_masks = split_masks_by_connected_component(refined_masks)
        # refined_masks = filter_masks_by_area(refined_masks, 1500*3)
    elif class_name == "splicing_connectors":
        refined_masks = smooth_masks(refined_masks)

    if len(refined_masks) == 0:
        if class_name == "pushpins" or class_name == "screw_bag":
            refined_masks = np.zeros((448,448), dtype=np.uint8)
        else:
            refined_masks = np.zeros((256,256), dtype=np.uint8)
    else:
        refined_masks = merge_masks(refined_masks)

    return refined_masks
    


def split_masks_by_connected_component(masks):
    result_masks = list()
    for i,mask in enumerate(masks):
        #mask_uint = turn_binary_to_int(mask['segmentation'])
        if type(mask) == dict:
            mask_uint = turn_binary_to_int(mask['segmentation'])
        else:
            mask_uint = mask
        # cv2.imshow("mask",mask_uint)
        # cv2.waitKey(0)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint,connectivity=4,ltype=None)
        for j in range(num_labels):
            area = stats[j,4]
            w,h = stats[j,2], stats[j,3]
            if w*h < mask_uint.shape[0]*mask_uint.shape[1]*0.95 and area>50:# and area>0.01*mask_uint.shape[0]*mask_uint.shape[1]:
            # if area > 10:
                #pass
                #result_masks.append({'area':area,"segmentation":labels==j})
                result_masks.append(turn_binary_to_int(labels==j))
    return result_masks

def color_masks(masks):
    # if type(masks) != list:
    #     masks = [masks]
    if type(masks) == list and len(masks) == 1:
        return np.where(masks[0],255,0).astype(np.uint8)
    if type(masks) != list and len(masks.shape) == 2:
        return np.where(masks!=0,255,0).astype(np.uint8)
    color_mask = np.zeros([masks[0].shape[0],masks[0].shape[1],3],dtype=np.uint8)
    masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        color_mask[mask!=0] = np.random.randint(0,255,[3])
    return color_mask

def merge_masks(masks, reverse=True):
    # remove empty masks
    masks = filter_small_masks(masks,threshold=0.0001)
    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    masks = sorted(masks,key=lambda x:np.sum(x),reverse = reverse)
    for i,mask in enumerate(masks):
        # mask = binary_fill_holes(mask).astype(np.uint8)
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(min(i+1, 255))
    return result_mask

def merge_masks_no_sort(masks):
    # remove empty masks
    masks = filter_small_masks(masks,threshold=0.0001)

    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    # masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        # mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(i+1)
    return result_mask

def split_masks_from_one_mask(masks):
    result_masks = list()
    result_idxs = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.0001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_sort(masks):
    """
    将多类别mask分割成单独的mask并按面积从小到大排序
    
    参数:
    masks: numpy数组，其中每个值代表一个区域编号
    
    返回:
    result_masks: 按面积从小到大排序的mask列表
    result_idxs: 与result_masks对应的原始标签索引
    """
    # 存储mask、原始索引和面积的信息
    mask_info = []
    
    for i in range(1, np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        area = np.sum(mask!=0)
        
        # 只保留面积比例大于阈值的mask
        if area/mask.size > 0.0001:
            mask_info.append((mask, i, area))
    
    # 按面积从小到大排序
    mask_info.sort(key=lambda x: x[2])
    
    # 分离排序后的mask和索引
    result_masks = [info[0] for info in mask_info]
    result_idxs = [info[1] for info in mask_info]
    
    return result_masks, result_idxs

def split_masks_from_one_mask_with_bg(masks):
    result_masks = list()
    result_idxs = list()
    for i in range(0,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.0001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_with_bg_torch(masks):
    result_masks = list()
    result_idxs = list()
    H, W = masks.shape
    for i in range(1,torch.max(masks)+1):
        mask = torch.zeros_like(masks)
        mask[masks==i] = 255
        if torch.sum(mask!=0) / (H * W) > 0.001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_torch(masks):
    result_masks = list()
    H, W = masks.shape
    for i in range(1,torch.max(masks)+1):
        mask = torch.zeros_like(masks)
        mask[masks==i] = 255
        result_masks.append(mask)
        # if torch.sum(mask!=0) / (H * W) > 0.001:
        #     result_masks.append(mask)
    return result_masks


def split_masks_from_one_mask_torch_sort(masks):
    """
    将多类别mask分割成单独的mask并按面积从小到大排序
    
    参数:
    masks: Torch张量，形状为[H, W]，其中每个值代表一个区域编号
    
    返回:
    按面积从小到大排序的mask列表
    """
    result_masks = []
    H, W = masks.shape
    
    # 遍历所有可能的mask标签
    max_label = torch.max(masks).item()
    
    # 存储每个mask及其面积
    masks_with_areas = []
    
    for i in range(1, max_label+1):
        mask = torch.zeros_like(masks)
        mask[masks==i] = 255
        
        # 计算当前mask的面积
        area = torch.sum(mask > 0).item()
        
        # 只添加非空mask
        if area > 0:
            masks_with_areas.append((mask, area))
    
    # 按面积从小到大排序
    masks_with_areas.sort(key=lambda x: x[1])
    
    # 提取排序后的mask
    result_masks = [mask for mask, _ in masks_with_areas]
    
    return result_masks

def resize_mask(mask,image_size):
    mask = cv2.resize(mask,[image_size,image_size],interpolation=cv2.INTER_LINEAR)
    mask[mask>128]=255
    mask[mask<=128]=0
    return mask

def intersect_ratio(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    if intersection.sum() == 0:
        return 0
    ratio = np.sum(intersection)/min([np.sum(mask1!=0),np.sum(mask2!=0)])
    ratio = 0 if np.isnan(ratio) else ratio
    return ratio

def iou(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    union = np.logical_or(mask1,mask2)
    return np.sum(intersection)/np.sum(union)

def remove_duplicate_masks(masks, iou_threshold=0.9):
    # List to store indices of masks to keep
    keep_masks = []

    for i in range(len(masks)):
        is_duplicate = False
        for j in range(len(keep_masks)):
            combine = np.hstack([masks[i],masks[keep_masks[j]]])
            combine = cv2.resize(combine,[512,256])
            cv2.imshow(f"{iou(masks[i], masks[keep_masks[j]])}",combine)
            cv2.waitKey(0)
            if iou(masks[i], masks[keep_masks[j]]) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_masks.append(i)

    # Filter the original list of masks
    unique_masks = [masks[i] for i in keep_masks]
    return unique_masks

def filter_small_masks(masks,threshold=0.001):
    new_masks = list()
    for mask in masks:
        if np.sum(mask!=0)/mask.size > threshold:
            new_masks.append(mask)
    return new_masks


def filter_masks_by_grounding_mask(grounding_mask,masks):
    new_mask = list()
    for mask in masks:
        if intersect_ratio(grounding_mask,mask) > 0.9:
            new_mask.append(mask)
    return new_mask

def merge_nested_masks(masks, contain_threshold=0.95):
    masks = filter_small_masks(masks, threshold=0.0001)
    masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)  # 大的在前
    merged = [False] * len(masks)  # 标记哪些被合并了

    for i in range(len(masks)):
        if merged[i]:
            continue
        for j in range(i + 1, len(masks)):
            if merged[j]:
                continue
            intersection = np.logical_and(masks[i], masks[j])
            ratio_j_in_i = np.sum(intersection) / np.sum(masks[j] != 0)
            if ratio_j_in_i > contain_threshold:
                # 将小的合并进大的
                masks[i] = np.logical_or(masks[i], masks[j])
                merged[j] = True

    # 返回没有被合并掉的掩码
    return [mask for i, mask in enumerate(masks) if not merged[i]]


def filter_by_combine(masks):
    #masks = split_masks_from_one_mask(merge_masks(masks))
    
    masks = filter_small_masks(masks,threshold=0.0001)
    masks = sorted(masks,key=lambda x:np.sum(x)) # small to large
    # if len(masks) < 1:
    #     return masks
    combine_masks = np.zeros_like(masks[0])
    result_masks = list()
    wait_masks = list()
    for i,mask in enumerate(masks):
        if intersect_ratio(combine_masks,mask) < 0.9 or i == 0:
            combine_masks = np.logical_or(combine_masks,mask)
            result_masks.append(mask)
        else:
            wait_masks.append(mask)

    if len(wait_masks) != 0:
        for mask in wait_masks:
            ratio = np.sum(np.logical_and(combine_masks,mask))/np.sum(mask!=0)
            if ratio < 0.9:
                combine_masks = np.logical_or(combine_masks,mask)
                result_masks.append(mask)
    return result_masks


def segmentation(img_paths,save_path,use_grounding_filter=False,no_sam=True):
    for p in tqdm.tqdm(img_paths,desc="segmentation..."):
        image_name = '/'.join((p.split(".")[0]).split("/")[-3:])
        print(image_name)
        refined_masks = cv2.imread(f"{save_path}/{image_name}/grounding_mask.png",cv2.IMREAD_GRAYSCALE)
        refined_masks = split_masks_from_one_mask(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = split_masks_by_connected_component(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = filter_by_combine(refined_masks)
        refined_masks_color = color_masks(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = merge_masks(refined_masks)
        # cv2.imwrite(f"{save_path}/{image_name}/all_masks.png",merge_masks(raw_splited_masks))
        # cv2.imwrite(f"{save_path}/{image_name}/all_masks_color.png",color_masks(raw_splited_masks))
        cv2.imwrite(f"{save_path}/{image_name}/refined_masks.png",refined_masks)
        cv2.imwrite(f"{save_path}/{image_name}/refined_masks_color.png",refined_masks_color)


# 方法1: 形态学操作
def morphology_clean(mask):
    kernel = np.ones((5, 5), np.uint8)  # 根据毛刺大小调整核的大小
    # 开运算 (去除小物体)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 闭运算 (填充小洞)
    result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

# 方法2: 中值滤波
def median_filter_clean(mask):
    # 根据毛刺大小调整窗口大小
    return cv2.medianBlur(mask.astype(np.uint8), 5)

# 方法3: 高斯滤波+阈值处理
def gaussian_threshold_clean(mask):
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    # 对模糊后的图像进行阈值处理，恢复清晰边界
    _, result = cv2.threshold(blurred, 0.5, mask.max(), cv2.THRESH_BINARY)
    return result

# 方法4: 区域连通性分析
def connectivity_clean(mask):
    # 将mask转换为二值图像(如果不是的话)
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 寻找所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary)
    
    # 过滤掉小区域
    min_size = 50  # 可根据实际情况调整
    result = np.zeros_like(mask)
    
    # 从1开始，因为0是背景
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = mask[labels == i]
    
    return result


def fill(mask):
    # 创建副本用于填充
    mask_to_fill = mask.copy()

    # 将mask转换为适合flood fill的格式
    if len(mask.shape) == 3:  # 如果是彩色图像
        mask_to_fill = cv2.cvtColor(mask_to_fill, cv2.COLOR_BGR2GRAY)

    # 创建一个比原图大2像素的掩码（洪水填充需要）
    h, w = mask_to_fill.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    # 对每个连通区域应用洪水填充
    for i in range(0, h, 20):
        for j in range(0, w, 20):
            if mask_to_fill[i, j] > 0:  # 如果不是背景
                # 获取当前位置的值作为种子点的值
                seed_value = int(mask_to_fill[i, j])
                cv2.floodFill(mask_to_fill, flood_mask, (j, i), seed_value, 
                            loDiff=0, upDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)

    return mask_to_fill

# 综合方法 (结合以上方法)
def clean_mask(mask):
    # 先用区域连通性去除小区域
    # cleaned = morphology_clean(mask)
    cleaned = connectivity_clean(mask)
    # 再用形态学操作平滑边缘
    cleaned = morphology_clean(cleaned)
    cleaned = fill(cleaned)
    return cleaned

# 处理多类别mask
def clean_multiclass_mask(mask):
    unique_values = np.unique(mask)
    result = np.zeros_like(mask)
    
    # 跳过背景(通常为0)
    for val in unique_values[unique_values > 0]:
        # 提取单一类别
        binary_mask = (mask == val).astype(np.uint8)
        # 清洁这个类别的mask
        cleaned_binary = clean_mask(binary_mask)
        # 将此类别添加回结果(避免类别重叠)
        result[cleaned_binary > 0] = val
        
    return result

import copy

def process_masks_juice_bottle(masks):
    # 如果mask数量小于等于3，不处理
    if len(masks) <= 3:
        return masks
    
    # 计算每个mask的面积（255像素的数量）和质心
    areas = []
    centroids_y = []  # 质心的垂直坐标
    
    for mask in masks:
        # 计算面积
        area = np.sum(mask == 255)
        areas.append(area)
        
        # 计算质心
        if area > 0:
            # 找到所有值为255的像素坐标
            y_coords, x_coords = np.where(mask == 255)
            # 计算质心
            centroid_y = np.mean(y_coords)
            centroids_y.append(centroid_y)
        else:
            # 如果mask为空，将质心设为中间位置
            centroids_y.append(128)
    
    # 找到面积最大的mask的索引
    max_area_idx = np.argmax(areas)
    
    # 排除最大面积的mask
    remaining_indices = list(range(len(masks)))
    remaining_indices.remove(max_area_idx)
    
    # 在剩余的mask中找最靠上的（质心y坐标最小）
    top_mask_idx = remaining_indices[np.argmin([centroids_y[i] for i in remaining_indices])]
    
    # 在剩余的mask中找最靠下的（质心y坐标最大）
    bottom_mask_idx = remaining_indices[np.argmax([centroids_y[i] for i in remaining_indices])]
    
    # 创建结果列表，首先深拷贝所有mask
    result_masks = copy.deepcopy(masks)
    
    # 需要保留的mask索引
    keep_indices = [max_area_idx, top_mask_idx, bottom_mask_idx]
    
    # 将其他mask合并到面积最大的mask中
    for i in range(len(masks)):
        if i not in keep_indices:
            # 合并：如果原mask或要合并的mask中有255，结果就是255
            result_masks[max_area_idx] = np.maximum(result_masks[max_area_idx], masks[i])
    
    # 只保留三个mask：最大的、最靠上的和最靠下的
    return [result_masks[i] for i in keep_indices]


def find_highest_fruit(items_list):
    # 初始化要查找的水果及其值
    fruits = {"banana": -1, "cherry": -1, "orange": -1, "tangerine": -1}
    
    # 遍历列表
    for item in items_list:
        # 分离名称和值
        name_part = item.split("(")[0]# .split()[0]
        value_part = item.split("(")[1].rstrip(")")
        value = float(value_part)
        
        # 检查是否是我们要找的水果
        if name_part in fruits and value > fruits[name_part]:
            fruits[name_part] = value
    
    # 找出值最大的水果
    max_fruit = None
    max_value = -1
    
    for fruit, value in fruits.items():
        if value > max_value:
            max_value = value
            max_fruit = fruit
    
    # 检查是否找到任何水果
    if max_value == -1:
        return "no fruit"
    
    return max_fruit


def filter_masks_by_area(masks, area_threshold):
    """
    过滤掉面积大于阈值的 mask
    
    参数:
    masks: 包含多个 [256, 256] 形状的 numpy 数组的列表，值为 0 或 255
    area_threshold: 面积阈值，大于此值的 mask 将被过滤掉
    
    返回:
    filtered_masks: 过滤后的 mask 列表
    """
    filtered_masks = []
    
    for i, mask in enumerate(masks):
        # 计算 mask 的面积 (值为 255 的像素数量)
        area = np.sum(mask == 255)
        
        # 如果面积小于或等于阈值，则保留该 mask
        if area <= area_threshold:
            filtered_masks.append(mask)

    # print(f"过滤前 mask 数量: {len(masks)}")
    # print(f"过滤后 mask 数量: {len(filtered_masks)}")
    
    return filtered_masks

def filter_masks_below_area(masks, area_threshold):
    """
    过滤掉面积大于阈值的 mask
    
    参数:
    masks: 包含多个 [256, 256] 形状的 numpy 数组的列表，值为 0 或 255
    area_threshold: 面积阈值，大于此值的 mask 将被过滤掉
    
    返回:
    filtered_masks: 过滤后的 mask 列表
    """
    filtered_masks = []
    
    for i, mask in enumerate(masks):
        # 计算 mask 的面积 (值为 255 的像素数量)
        area = np.sum(mask == 255)
        
        # 如果面积小于或等于阈值，则保留该 mask
        if area >= area_threshold:
            filtered_masks.append(mask)

    # print(f"过滤前 mask 数量: {len(masks)}")
    # print(f"过滤后 mask 数量: {len(filtered_masks)}")
    
    return filtered_masks

def filter_masks_in_area(masks, min_area, max_area):
    """
    过滤掉面积大于阈值的 mask
    
    参数:
    masks: 包含多个 [256, 256] 形状的 numpy 数组的列表，值为 0 或 255
    area_threshold: 面积阈值，大于此值的 mask 将被过滤掉
    
    返回:
    filtered_masks: 过滤后的 mask 列表
    """
    filtered_masks = []
    
    for i, mask in enumerate(masks):
        # 计算 mask 的面积 (值为 255 的像素数量)
        area = np.sum(mask == 255)
        
        # 如果面积小于或等于阈值，则保留该 mask
        if area > max_area or area < min_area :
            filtered_masks.append(mask)
            
    # print(f"过滤前 mask 数量: {len(masks)}")
    # print(f"过滤后 mask 数量: {len(filtered_masks)}")
    
    return filtered_masks

# 函数：将线段表示为中点和方向向量
def line_to_vector(line):
    x1, y1, x2, y2 = line
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx, dy = dx / length, dy / length
    return (mid_x, mid_y, dx, dy, length)

def has_significant_gap(line1, line2, horizontal=True, gap_threshold=40):
    x1a, y1a, x2a, y2a = line1
    x1b, y1b, x2b, y2b = line2
    
    if horizontal:
        # 对于水平线，检查x方向上是否有明显间隔
        max_x1 = max(x1a, x2a)
        min_x1 = min(x1a, x2a)
        max_x2 = max(x1b, x2b)
        min_x2 = min(x1b, x2b)
        
        # 线段间没有重叠且有足够间隔
        return (min_x2 - max_x1 > gap_threshold) or (min_x1 - max_x2 > gap_threshold)
    else:
        # 对于垂直线，检查y方向上是否有明显间隔
        max_y1 = max(y1a, y2a)
        min_y1 = min(y1a, y2a)
        max_y2 = max(y1b, y2b)
        min_y2 = min(y1b, y2b)
        
        # 线段间没有重叠且有足够间隔
        return (min_y2 - max_y1 > gap_threshold) or (min_y1 - max_y2 > gap_threshold)

# 改进的合并函数
def merge_similar_lines(lines, distance_threshold=50, angle_threshold=0.2, gap_threshold=30 * 2, horizontal=True):
    if len(lines) <= 1:
        return lines
    
    # 根据线条的位置进行排序（水平线按y排序，垂直线按x排序）
    if horizontal:
        sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)  # 按y坐标排序
    else:
        sorted_lines = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)  # 按x坐标排序

    sorted_lines = lines
    
    # 提取线段参数
    line_vectors = [line_to_vector(line) for line in sorted_lines]
    merged_lines = []
    used = [False] * len(sorted_lines)
    
    for i, (mx1, my1, dx1, dy1, l1) in enumerate(line_vectors):
        if used[i]:
            continue
            
        # 当前线段
        current_line = sorted_lines[i]
        used[i] = True
        current_group = [current_line]
        
        for j, (mx2, my2, dx2, dy2, l2) in enumerate(line_vectors):
            if used[j] or i == j:
                continue
                
            # 检查方向是否相似
            angle_sim = abs(dx1*dx2 + dy1*dy2)  # 方向向量的点积
            
            # 计算中点之间的距离
            if horizontal:
                # 对于水平线，主要比较y坐标的差异
                dist = abs(my1 - my2)
            else:
                # 对于垂直线，主要比较x坐标的差异
                dist = abs(mx1 - mx2)
            
            # 检查是否有显著的间隔
            has_gap = has_significant_gap(current_line, sorted_lines[j], horizontal, gap_threshold)
            
            # 如果线条足够接近、方向相似且没有显著间隔，则合并
            if dist < distance_threshold and angle_sim > (1 - angle_threshold) and not has_gap:
                current_group.append(sorted_lines[j])
                used[j] = True
        
        # 合并当前组中的所有线段
        if len(current_group) > 0:
            x_coords = []
            y_coords = []
            for x1, y1, x2, y2 in current_group:
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            
            if horizontal:
                # 水平线：选择最左和最右的x坐标，y取平均
                avg_y = sum(y_coords) / len(y_coords)
                min_x = min(x_coords)
                max_x = max(x_coords)
                merged_line = [min_x, avg_y, max_x, avg_y]
            else:
                # 垂直线：选择最上和最下的y坐标，x取平均
                avg_x = sum(x_coords) / len(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)
                merged_line = [avg_x, min_y, avg_x, max_y]
                
            merged_lines.append(merged_line)
    
    return merged_lines


def find_lines(image):

    # 转换到灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用自适应阈值化，突出白色线条
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)

    # 使用形态学操作来增强线条
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(
        255 - morph,  # 反转图像使线条为白色
        rho=1,
        theta=np.pi/180,
        threshold=80 * 2,
        minLineLength=30 * 2,
        maxLineGap=10 * 2
    )

    # 分离水平线和垂直线
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线的角度
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # 根据角度区分水平线和垂直线
            if angle < 2 or angle > 178:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 88 < angle < 92:
                vertical_lines.append((x1, y1, x2, y2))

    # 合并相似的线条
    merged_horizontal = merge_similar_lines(horizontal_lines, distance_threshold=30 * 2, horizontal=True)
    merged_horizontal = merge_similar_lines(merged_horizontal, distance_threshold=30 * 2, horizontal=True)
    merged_vertical = merge_similar_lines(vertical_lines, distance_threshold=30 * 2, horizontal=False)
    merged_vertical = merge_similar_lines(merged_vertical, distance_threshold=30 * 2, horizontal=False)

    return merged_horizontal, merged_vertical


def check_pin_distribution(pin_centers, horizontal_lines, vertical_lines, image=None):
    """
    检查图钉是否在网格中均匀分布，每个方格刚好有一个图钉
    
    参数:
    - pin_centers: 图钉质心的坐标列表，每个元素为 (x, y) 元组
    - horizontal_lines: 水平线的坐标，每个元素为 [x1, y1, x2, y2]
    - vertical_lines: 垂直线的坐标，每个元素为 [x1, y1, x2, y2]
    - image: 可选，用于可视化的原始图像
    
    返回:
    - 布尔值，表示是否每个方格刚好有一个图钉
    - 每个方格包含的图钉数量的二维数组
    - 可视化结果图像（如果提供了image参数）
    """
    # 确保水平线和垂直线按位置排序
    horizontal_lines = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    # 从水平线和垂直线提取网格边界
    h_positions = [(line[1] + line[3]) / 2 for line in horizontal_lines]
    v_positions = [(line[0] + line[2]) / 2 for line in vertical_lines]
    
    # 确保我们有足够的线来构成3x5的网格
    if len(h_positions) < 4 or len(v_positions) < 6:
        return False, None, None
    
    # 创建网格边界（包括图像边界）
    top_bound = 0
    bottom_bound = image.shape[0] if image is not None else 1000
    left_bound = 0
    right_bound = image.shape[1] if image is not None else 1000
    
    # 定义行和列的边界
    row_bounds = h_positions # [top_bound] + h_positions + [bottom_bound]
    col_bounds = v_positions #[left_bound] + v_positions + [right_bound]
    
    # 初始化网格中的图钉计数
    grid_counts = np.zeros((len(row_bounds)-1, len(col_bounds)-1), dtype=int)
    
    # 确定每个图钉所在的方格
    pin_locations = []  # 存储每个图钉所在的方格坐标 (row, col)
    
    for pin_y, pin_x in pin_centers:
        # 找到图钉所在的行
        row_idx = None
        for i in range(len(row_bounds) - 1):
            if row_bounds[i] <= pin_y < row_bounds[i+1]:
                row_idx = i
                break
        
        # 找到图钉所在的列
        col_idx = None
        for j in range(len(col_bounds) - 1):
            if col_bounds[j] <= pin_x < col_bounds[j+1]:
                col_idx = j
                break
        
        # 如果图钉在网格内，增加相应方格的计数
        if row_idx is not None and col_idx is not None:
            grid_counts[row_idx, col_idx] += 1
            pin_locations.append((row_idx, col_idx))
        else:
            # 如果有图钉不在任何方格内，返回False
            return False, grid_counts, None
    
    # 检查每个方格是否刚好有一个图钉
    result = np.all(grid_counts == 1)
    
    # 如果提供了图像，则可视化结果
    visualization = None
    if image is not None:
        visualization = image.copy()
        
        # 绘制网格
        for h_line in horizontal_lines:
            x1, y1, x2, y2 = map(int, h_line)
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        
        for v_line in vertical_lines:
            x1, y1, x2, y2 = map(int, v_line)
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
        
        # 绘制图钉位置
        for (pin_x, pin_y), (row, col) in zip(pin_centers, pin_locations):
            color = 'blue' if grid_counts[row, col] == 1 else 'red'
            plt.plot(pin_x, pin_y, 'o', color=color, markersize=8)
            
            # 在每个方格中标注图钉数量
            cell_center_x = (col_bounds[col] + col_bounds[col+1]) / 2
            cell_center_y = (row_bounds[row] + row_bounds[row+1]) / 2
            plt.text(cell_center_x, cell_center_y, str(grid_counts[row, col]), 
                     color='white', ha='center', va='center', fontsize=12)
        
        plt.title(f"{'每个方格刚好有一个图钉' if result else '图钉分布不均匀'}")
        plt.axis('off')
    
    return result, grid_counts, visualization


def get_mask_bounds(mask):
    """返回 mask 的边界框 (top, bottom, left, right)"""
    ys, xs = np.where(mask > 0)
    return ys.min(), ys.max(), xs.min(), xs.max()

def dilate_mask(mask, kernel_size=3):
    """对 mask 进行膨胀操作，使得相邻 mask 可以相交"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def get_relative_connection_y(other_mask, middle_mask):
    """
    计算 other_mask 与 middle_mask 的交点在 other_mask 中的相对垂直位置（归一化 y）。
    """
    intersection = cv2.bitwise_and(dilate_mask(other_mask), dilate_mask(middle_mask))
    ys, xs = np.where(intersection > 0)
    if len(ys) == 0:
        return None  # 无连接

    y_connect = np.mean(ys)  # 连接点的平均 y 坐标
    y_top, y_bottom, _, _ = get_mask_bounds(other_mask)
    rel_y = (y_connect - y_top) / (y_bottom - y_top + 1e-5)  # 归一化到 0~1
    return rel_y

def merge_small_regions_to_smallest_neighbor_not_circle(mask_list, min_area_threshold=2000, max_target_area_threshold=6000,
                                             merge_to_kth_largest=-1, circularity_threshold=0.8):  # NEW param
    current_masks = [mask.copy() for mask in mask_list]
    iteration = 0
    max_iterations = 5
    height, width = current_masks[0].shape
    initial_mask_areas = [np.sum(mask > 0) for mask in current_masks]
    
    excluded_targets = set()
    if max_target_area_threshold is not None:
        excluded_targets = {i for i, area in enumerate(initial_mask_areas) if area > max_target_area_threshold}

    while iteration < max_iterations:
        mask_areas = [np.sum(mask > 0) for mask in current_masks]
        small_mask_indices = [i for i, area in enumerate(mask_areas) if area < min_area_threshold]
        
        if len(small_mask_indices) == 0:
            break

        volume = np.zeros((height, width), dtype=np.int32)
        for i, mask in enumerate(current_masks):
            mask_region = (mask > 0)
            volume = np.where(volume == 0, (i + 1) * mask_region, volume)
        
        processed_masks = set()
        any_merges = False
        small_areas_with_indices = [(mask_areas[i], i) for i in small_mask_indices]
        sorted_small_indices = [idx for _, idx in sorted(small_areas_with_indices, reverse=True)]

        for idx in sorted_small_indices:
            if idx in processed_masks:
                continue
            
            small_mask = (volume == (idx + 1))
            if not np.any(small_mask):
                processed_masks.add(idx)
                continue

            # --- NEW: Check for circularity ---
            labeled_mask = measure.label(small_mask.astype(np.uint8))
            props = measure.regionprops(labeled_mask)
            if props:
                region = props[0]
                area = region.area
                perimeter = region.perimeter if region.perimeter > 0 else 1
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity >= circularity_threshold:
                    # This small mask is circular enough — do not merge
                    processed_masks.add(idx)
                    continue
            # --- END NEW ---

            area_now = np.sum(small_mask)
            dilated = morphology.binary_dilation(small_mask)
            neighbor_mask = dilated & ~small_mask
            neighbor_labels = np.unique(volume[neighbor_mask])
            neighbor_labels = [l for l in neighbor_labels if l != 0 and l != (idx + 1)]
            neighbor_labels = [l for l in neighbor_labels if (l - 1) not in excluded_targets]

            if neighbor_labels:
                if merge_to_kth_largest > 0:
                    neighbor_areas = [(l, mask_areas[l - 1]) for l in neighbor_labels]
                    neighbor_areas.sort(key=lambda x: x[1])
                    k_idx = min(merge_to_kth_largest - 1, len(neighbor_areas) - 1)
                    target_label = neighbor_areas[k_idx][0]
                elif merge_to_kth_largest == -1:
                    target_label = max(neighbor_labels, key=lambda l: mask_areas[l - 1])
                else:
                    target_label = min(neighbor_labels, key=lambda l: mask_areas[l - 1])
                
                volume[small_mask] = target_label
                mask_areas[target_label - 1] += mask_areas[idx]
                mask_areas[idx] = 0
                processed_masks.add(idx)
                any_merges = True

        unique_labels = np.unique(volume)
        unique_labels = unique_labels[unique_labels > 0]
        new_masks = [(volume == label).astype(np.uint8) * 255 for label in unique_labels]
        current_masks = new_masks
        iteration += 1
        print(f"迭代 {iteration}: 剩余 {len(current_masks)} 个mask")

    if iteration == max_iterations:
        print(f"警告: 达到最大迭代次数 ({max_iterations})，可能仍有小于阈值的mask。")

    return current_masks

def filter_nut_and_bolt_masks(mask_list, circularity_threshold=0.75, aspect_ratio_threshold=3.0):
    """
    保留圆形（螺母）和长条形（螺钉）mask，过滤掉其他形状的mask

    参数:
    - mask_list: list of [H, W] 二值掩码，每个元素为0或255
    - circularity_threshold: 判定为圆形的圆形度下限（建议 0.75 ~ 0.9）
    - aspect_ratio_threshold: 判定为长条形的长宽比下限（建议 >= 3.0）

    返回:
    - 过滤后的mask列表，只包含圆形或长条形区域
    """
    filtered_masks = []

    for mask in mask_list:
        binary = (mask > 0).astype(np.uint8)
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)

        if not props:
            continue

        region = props[0]
        area = region.area
        perimeter = region.perimeter if region.perimeter > 0 else 1
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # 长宽比
        if region.minor_axis_length == 0:
            aspect_ratio = np.inf  # 线段
        else:
            aspect_ratio = region.major_axis_length / region.minor_axis_length

        is_circular = circularity >= circularity_threshold
        is_long_bar = aspect_ratio >= aspect_ratio_threshold

        if is_circular or is_long_bar:
            filtered_masks.append(mask)

    return filtered_masks
# def extract_mask_features(mask, original_image=None):
#     """
#     提取掩码图像中各个数值区域的特征
    
#     参数:
#     mask: numpy数组，包含不同数值的掩码图像
#     original_image: numpy数组，原始图像(可选)，用于计算每个区域的平均颜色
    
#     返回:
#     字典，包含每个标签区域的特征
#     """
#     # 获取mask中的唯一值（排除背景0值）
#     unique_values = np.unique(mask)
    
#     # 创建一个字典来存储每个mask值的特征
#     features = {}
    
#     # 遍历所有可能的值（从0到mask中的最大值）
#     max_possible_value = 4
#     for value in range(1, int(max_possible_value) + 1):
#         # 检查这个值是否存在于mask中
#         if value in unique_values:
#             # 为当前值创建二值掩码
#             binary_mask = (mask == value).astype(np.uint8)
            
#             # 计算区域面积（像素数）
#             area = np.sum(binary_mask)
            
#             # 找到区域的边界框
#             rows, cols = np.where(binary_mask > 0)
#             if len(rows) > 0 and len(cols) > 0:
#                 min_row, max_row = np.min(rows), np.max(rows)
#                 min_col, max_col = np.min(cols), np.max(cols)
#                 bounding_box = {
#                     'min_row': int(min_row),
#                     'max_row': int(max_row), 
#                     'min_col': int(min_col), 
#                     'max_col': int(max_col),
#                     'width': int(max_col - min_col + 1),
#                     'height': int(max_row - min_row + 1)
#                 }
#             else:
#                 bounding_box = None
            
#             # 计算质心
#             if area > 0:
#                 centroid = ndimage.measurements.center_of_mass(binary_mask)
#                 centroid = (float(centroid[0]), float(centroid[1]))
#             else:
#                 centroid = None
            
#             # 计算平均颜色（如果提供了原始图像）
#             avg_color = None
#             if original_image is not None:
#                 if binary_mask.sum() > 0:
#                     if len(original_image.shape) == 3:  # 彩色图像
#                         avg_color = []
#                         for channel in range(original_image.shape[2]):
#                             channel_avg = np.mean(original_image[:,:,channel][binary_mask > 0])
#                             avg_color.append(float(channel_avg))
#                     else:  # 灰度图像
#                         avg_color = float(np.mean(original_image[binary_mask > 0]))
            
#             # 计算区域的最小外接圆半径
#             labeled_mask, num_features = ndimage.label(binary_mask)
#             props = measure.regionprops(labeled_mask)
#             equivalent_diameter = 0
#             if props:
#                 equivalent_diameter = float(props[0].equivalent_diameter)
            
#             # 存储特征
#             features[value] = {
#                 'exists': True,
#                 'area': int(area),
#                 # 'bounding_box': bounding_box,
#                 'centroid': centroid,
#                 'avg_color': avg_color,
#                 # 'equivalent_diameter': equivalent_diameter
#             }
#         else:
#             # 记录不存在的值
#             features[value] = {'exists': False}
    
#     return features


def extract_mask_features_old(mask_list, original_image=None):
    """
    提取掩码列表中各个mask区域的特征
    
    参数:
    mask_list: 长度为k的列表，每个元素是[256, 256]形状的tensor，值为0或255
    original_image: numpy数组或tensor，原始图像(可选)，用于计算每个区域的平均颜色
    
    返回:
    字典，包含每个mask区域的特征
    """
    # 创建一个字典来存储每个mask的特征
    features = {}
    
    # 将原始图像转换为numpy数组(如果是tensor)
    if original_image is not None and torch.is_tensor(original_image):
        if original_image.device != torch.device('cpu'):
            original_image = original_image.cpu()
        original_image = original_image.numpy()
        
        # 如果原始图像是[C, H, W]格式，转换为[H, W, C]格式
        if len(original_image.shape) == 3 and original_image.shape[0] == 3:
            original_image = np.transpose(original_image, (1, 2, 0))
    
    # 遍历mask列表
    for i, mask_tensor in enumerate(mask_list):
        # 将mask转换为numpy数组
        if torch.is_tensor(mask_tensor):
            if mask_tensor.device != torch.device('cpu'):
                mask_tensor = mask_tensor.cpu()
            mask = mask_tensor.numpy()
        else:
            mask = mask_tensor
        
        # 创建二值掩码 (255 -> 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 计算区域面积（像素数）
        area = np.sum(binary_mask)
        
        # 如果区域为空，跳过
        if area == 0:
            features[i] = {'exists': False}
            continue
        
        # 找到区域的边界框
        rows, cols = np.where(binary_mask > 0)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        bounding_box = {
            'min_row': int(min_row),
            'max_row': int(max_row), 
            'min_col': int(min_col), 
            'max_col': int(max_col),
            'width': int(max_col - min_col + 1),
            'height': int(max_row - min_row + 1)
        }
        
        # 计算质心
        centroid = ndimage.measurements.center_of_mass(binary_mask)
        centroid = (float(centroid[0]), float(centroid[1]))
        
        # 计算平均颜色（如果提供了原始图像）
        avg_color = None
        if original_image is not None:
            if len(original_image.shape) == 3:  # 彩色图像
                avg_color = []
                for channel in range(original_image.shape[2]):
                    channel_avg = np.mean(original_image[:,:,channel][binary_mask > 0])
                    avg_color.append(float(channel_avg))
            else:  # 灰度图像
                avg_color = float(np.mean(original_image[binary_mask > 0]))
        
        # 计算区域的属性
        labeled_mask, num_features_label = ndimage.label(binary_mask)
        props = measure.regionprops(labeled_mask)
        
        # 提取常用属性
        properties = {}
        if props:
            # 等效直径 = 2 * √(面积/π)
            properties['equivalent_diameter'] = float(props[0].equivalent_diameter)
            # 周长
            properties['perimeter'] = float(props[0].perimeter) if hasattr(props[0], 'perimeter') else 0
            # 离心率 (0表示圆形，1表示线段)
            properties['eccentricity'] = float(props[0].eccentricity) if hasattr(props[0], 'eccentricity') else 0
            # 密实度 (区域面积与凸包面积之比)
            properties['solidity'] = float(props[0].solidity) if hasattr(props[0], 'solidity') else 0
        
        # 存储特征
        features[i] = {
            'area': int(area),
            # 'bounding_box': bounding_box,
            'centroid': centroid,
            'avg_color': avg_color,
            # 'properties': properties
        }
    
    return features


def extract_mask_features(mask_list, original_image=None, class_name=None):
    """
    提取掩码列表中各个mask区域的特征
    
    参数:
    mask_list: 长度为k的列表，每个元素是[256, 256]形状的tensor，值为0或255
    original_image: numpy数组或tensor，原始图像(可选)，用于计算每个区域的平均颜色
    
    返回:
    字典，包含每个mask区域的特征
    """
    # 创建一个字典来存储每个mask的特征
    features = {}
    
    # 将原始图像转换为numpy数组(如果是tensor)
    if original_image is not None and torch.is_tensor(original_image):
        if original_image.device != torch.device('cpu'):
            original_image = original_image.cpu()
        original_image = original_image.numpy()
        
        # 如果原始图像是[C, H, W]格式，转换为[H, W, C]格式
        if len(original_image.shape) == 3 and original_image.shape[0] == 3:
            original_image = np.transpose(original_image, (1, 2, 0))
    
    # 遍历mask列表
    for i, mask_tensor in enumerate(mask_list):
        # 将mask转换为numpy数组
        if torch.is_tensor(mask_tensor):
            if mask_tensor.device != torch.device('cpu'):
                mask_tensor = mask_tensor.cpu()
            mask = mask_tensor.numpy()
        else:
            mask = mask_tensor
        
        # 创建二值掩码 (255 -> 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 计算区域面积（像素数）
        area = np.sum(binary_mask)
        
        # 如果区域为空，跳过
        if area == 0:
            features[i] = {'exists': False}
            continue
        
        # 找到区域的边界框
        rows, cols = np.where(binary_mask > 0)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        bounding_box = {
            'min_row': int(min_row),
            'max_row': int(max_row), 
            'min_col': int(min_col), 
            'max_col': int(max_col),
            'width': int(max_col - min_col + 1),
            'height': int(max_row - min_row + 1)
        }
        
        # 计算质心
        centroid = ndimage.measurements.center_of_mass(binary_mask)
        centroid = (float(centroid[0]), float(centroid[1]))
        
        # 计算平均颜色（如果提供了原始图像）
        avg_color = None
        if original_image is not None:
            if len(original_image.shape) == 3:  # 彩色图像
                avg_color = []
                for channel in range(original_image.shape[2]):
                    channel_avg = np.mean(original_image[:,:,channel][binary_mask > 0])
                    avg_color.append(float(channel_avg))
            else:  # 灰度图像
                avg_color = float(np.mean(original_image[binary_mask > 0]))
        
        # 计算区域的属性
        labeled_mask, num_features_label = ndimage.label(binary_mask)
        props = measure.regionprops(labeled_mask)
        
        # 提取常用属性
        properties = {}
        if props:
            # 等效直径 = 2 * √(面积/π)
            properties['equivalent_diameter'] = float(props[0].equivalent_diameter)
            # 周长
            properties['perimeter'] = float(props[0].perimeter) if hasattr(props[0], 'perimeter') else 0
            # 离心率 (0表示圆形，1表示线段)
            properties['eccentricity'] = float(props[0].eccentricity) if hasattr(props[0], 'eccentricity') else 0
            # 密实度 (区域面积与凸包面积之比)
            properties['solidity'] = float(props[0].solidity) if hasattr(props[0], 'solidity') else 0
        
        # 计算宽度乘以1.7后的外接圆直径 - 改进版方法
        # 获取所有非零像素的坐标
        # points = np.column_stack(np.where(binary_mask > 0))
        
        if (class_name == "pushpins" or class_name == "screw_bag") and area > 0:
            # 找到掩码轮廓
            contours = measure.find_contours(binary_mask, 0.5)
            
            enclosing_circle_diameter = 0
            
            if contours:
                # 使用最大的轮廓
                contour = max(contours, key=len)
                
                # 对轮廓点应用宽度缩放
                contour_scaled = contour.copy()
                if class_name == "pushpins":
                    contour_scaled[:, 1] *= 1.7  # 第1列是x坐标
                if class_name == "screw_bag":
                    contour_scaled[:, 1] *= 1.5
                
                # 找到轮廓的凸包，进一步减少点数
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(contour_scaled)
                    hull_points = contour_scaled[hull.vertices]
                    
                    # 计算凸包点之间的最大距离，这通常是外接圆直径的良好近似
                    max_dist_squared = 0
                    n_points = len(hull_points)
                    
                    # 如果点数仍然很多，可以采样减少计算量
                    if n_points > 100:
                        # 均匀采样，保留约50个点
                        step = max(1, n_points // 50)
                        hull_points = hull_points[::step]
                        n_points = len(hull_points)
                    
                    for k in range(n_points):
                        for j in range(k+1, n_points):
                            dist_squared = np.sum((hull_points[k] - hull_points[j])**2)
                            if dist_squared > max_dist_squared:
                                max_dist_squared = dist_squared
                    
                    enclosing_circle_diameter = float(np.sqrt(max_dist_squared))
                    
                except (ImportError, ValueError):
                    # 如果ConvexHull计算失败，回退到轮廓点的边界框方法
                    min_row, max_row = np.min(contour_scaled[:, 0]), np.max(contour_scaled[:, 0])
                    min_col, max_col = np.min(contour_scaled[:, 1]), np.max(contour_scaled[:, 1])
                    height_span = max_row - min_row
                    width_span = max_col - min_col
                    enclosing_circle_diameter = float(np.sqrt(height_span**2 + width_span**2))
            else:
                # 如果找不到轮廓，使用边界框方法
                height_span = bounding_box['height']
                width_span = bounding_box['width'] * 1.7  # 宽度乘以1.7
                enclosing_circle_diameter = float(np.sqrt(height_span**2 + width_span**2))
        else:
            enclosing_circle_diameter = 0.0
        
        # 存储特征
        features[i] = {
            'area': int(area),
            # 'bounding_box': bounding_box,
            'centroid': centroid,
            'avg_color': avg_color,
            # 'properties': properties,
            'enclosing_circle_diameter': enclosing_circle_diameter  # 新增特征
        }
    
    return features

def get_principal_angle(mask):
    """返回主轴相对于水平线的角度，使其旋转后竖直"""
    ys, xs = np.where(mask > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)  # 转为 float 类型
    coords -= np.mean(coords, axis=0)
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg - 90

def rotate_point(x, y, angle_deg, center):
    """将点 (x, y) 绕 center 逆时针旋转 angle_deg 度"""
    angle_rad = np.radians(angle_deg)
    ox, oy = center
    x_rot = ox + np.cos(angle_rad) * (x - ox) - np.sin(angle_rad) * (y - oy)
    y_rot = oy + np.sin(angle_rad) * (x - ox) + np.cos(angle_rad) * (y - oy)
    return x_rot, y_rot


import os
import json

def get_relative_connection_y_by_rotating_mask(other_mask, middle_mask):
    """
    在原始图中找到交点，然后将 other_mask 和交点坐标一起旋转，再计算竖直方向的相对 y。
    """
    # 1. 找到交点
    intersection = cv2.bitwise_and(dilate_mask(other_mask), dilate_mask(middle_mask))
    ys, xs = np.where(intersection > 0)
    if len(xs) == 0:
        return None
    x_connect, y_connect = np.mean(xs), np.mean(ys)

    # 2. 计算主方向并旋转角度
    angle = get_principal_angle(other_mask)

    # 3. 计算旋转中心
    center = tuple(np.array(other_mask.shape[::-1]) / 2)

    # 4. 旋转 other_mask
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_other = cv2.warpAffine(other_mask.astype(np.uint8), rot_mat, other_mask.shape[::-1], flags=cv2.INTER_NEAREST)

    # 5. 旋转连接点坐标
    x_rot, y_rot = rotate_point(x_connect, y_connect, angle, center)

    # 6. 计算旋转后 mask 的 y 范围
    ys_mask = np.where(rotated_other > 0)[0]
    if len(ys_mask) == 0:
        return None
    y_top, y_bottom = np.min(ys_mask), np.max(ys_mask)

    # 7. 归一化 y
    rel_y = (y_rot - y_top) / (y_bottom - y_top + 1e-5)
    return rel_y

def compute_logical_score(masks, class_name, image_idx):

    logical_features = extract_mask_features(masks, cv2.imread(f"src/eval/submission/test_images/{class_name}/{str(image_idx)}/image_0.jpg"),class_name)

    logical_score = 0

    if class_name == "breakfast_box":

        if logical_features[0]['avg_color'][0] < 43 or logical_features[0]['avg_color'][0] > 63.5 or logical_features[0]['avg_color'][1] < 67 or logical_features[0]['avg_color'][1] > 104 or logical_features[0]['avg_color'][2] < 136.5 or logical_features[0]['avg_color'][2] > 207.5:
            logical_score += 1

        if logical_features[1]['avg_color'][0] < 42 or logical_features[1]['avg_color'][0] > 76 or logical_features[1]['avg_color'][1] < 43.5 or logical_features[1]['avg_color'][1] > 131.5 or logical_features[1]['avg_color'][2] < 99 or logical_features[1]['avg_color'][2] > 202.5:
            logical_score += 1

        if logical_features[2]['avg_color'][0] < 42 or logical_features[2]['avg_color'][0] > 81 or logical_features[2]['avg_color'][1] < 41 or logical_features[2]['avg_color'][1] > 148 or logical_features[2]['avg_color'][2] < 87 or logical_features[2]['avg_color'][2] > 205.5:
            logical_score += 1

        if (logical_features[0]['centroid'][1] + logical_features[1]['centroid'][1] + logical_features[2]['centroid'][1]) / 3 > 128:
            print("012 center")
            logical_score += 1

        if (logical_features[3]['avg_color'][2]) > 160 or (logical_features[3]['avg_color'][2]) < 90:
            print("3 color")
            logical_score += 1

        if logical_features[3]['area'] < 6000 or logical_features[3]['centroid'][0] < 128:
            print("3 area")
            logical_score += 1

        if logical_features[2]['area'] > 5000:
            print("2 area")
            logical_score += 1

        if (logical_features[3]['area'] + logical_features[4]['area']) / 2 < 10000 or logical_features[5]['area'] > 29000 or (logical_features[3]['area'] + logical_features[4]['area']) / 2 > 12000:
            print("34 area")
            logical_score += 1
        
        with open(f"src/eval/submission/test_masks/{class_name}/{str(image_idx)}/pred_phrases.json") as f:
            items = json.load(f)
            orange_count = 0
            banana_count = 0
            almond_count = 0
            for item in items:
                score = float(item.split("(")[-1].replace(")",""))
                if "orange" in item.split()[0] and score > 0.3:
                    orange_count += 1
                if "banana" in item:
                    banana_count += 1
                if "almond" in item:
                    almond_count += 1

            if orange_count != 2:
                print("orange count")
                logical_score += 1

    if class_name == "juice_bottle":

        if logical_features[1]['area'] < 1600 or logical_features[1]['area'] > 1900:
                    logical_score += 1
        if logical_features[2]['area'] < 3000 or logical_features[2]['area'] > 3700:
            logical_score += 1

        if logical_features[1]['avg_color'][0] < 75 or logical_features[1]['avg_color'][0] > 128 or logical_features[1]['avg_color'][1] < 144 or logical_features[1]['avg_color'][1] > 182 or logical_features[1]['avg_color'][2] < 176 or logical_features[1]['avg_color'][2] > 206.5:
            logical_score += 1
        
        if logical_features[2]['avg_color'][0] < 77 or logical_features[2]['avg_color'][0] > 123 or logical_features[2]['avg_color'][1] < 175 or logical_features[2]['avg_color'][1] > 222 or logical_features[2]['avg_color'][2] < 220 or logical_features[2]['avg_color'][2] > 248:
            logical_score += 1

        stand_banana_color = [97, 109, 113.5]
        stand_orange_color = [56, 103, 117]
        stand_cherry_color = [34, 37, 63.5]

        with open(f"src/eval/submission/test_masks/{class_name}/{str(image_idx)}/pred_phrases.json") as f:
            items = json.load(f)
            fruit = find_highest_fruit(items)

            if fruit == "no fruit":
                print("no fruit")
                logical_score += 1
            elif fruit == "banana":
                if abs(logical_features[3]['avg_color'][0] - stand_banana_color[0]) > 7 or abs(logical_features[3]['avg_color'][1] - stand_banana_color[1]) > 12 or abs(logical_features[3]['avg_color'][2] - stand_banana_color[2]) > 14.5:
                    print("banana color")
                    logical_score += 1
            elif fruit == "orange" or fruit == "tangerine":
                if abs(logical_features[3]['avg_color'][0] - stand_orange_color[0]) > 4 or abs(logical_features[3]['avg_color'][1] - stand_orange_color[1]) > 8 or abs(logical_features[3]['avg_color'][2] - stand_orange_color[2]) > 9:
                    print("orange color")
                    logical_score += 1
            elif fruit == "cherry":
                if abs(logical_features[3]['avg_color'][0] - stand_cherry_color[0]) > 3 or abs(logical_features[3]['avg_color'][1] - stand_cherry_color[1]) > 3 or abs(logical_features[3]['avg_color'][2] - stand_cherry_color[2]) > 3.5:
                    print("cherry color")
                    logical_score += 1

        if logical_features[1]['area'] > 2500:
            print("label area")
            logical_score += 1

        if logical_features[1]['centroid'][0] > 220 or logical_features[1]['centroid'][0] < 198:
            print("label position")
            logical_score += 1

        if logical_features[2]['centroid'][0] < 125:
            print("label position")
            logical_score += 1

        if abs(logical_features[0]['centroid'][0] - logical_features[2]['centroid'][0]) + abs(logical_features[0]['centroid'][1] - logical_features[2]['centroid'][1]) >= 10:
            print("fruit position")
            logical_score += 1

    if class_name == "pushpins":

        stand_pushpins_color = [50, 125.5, 168]

        if logical_features[0]['enclosing_circle_diameter'] < 80 or logical_features[1]['enclosing_circle_diameter'] < 80:
            print("pin length")
            logical_score += 1

        for i in range(len(logical_features)):
            if abs(logical_features[i]['avg_color'][0] - stand_pushpins_color[0]) > 9 or abs(logical_features[i]['avg_color'][1] - stand_pushpins_color[1]) > 28.5 or abs(logical_features[i]['avg_color'][2] - stand_pushpins_color[2]) > 29:
                print("pin color")
                logical_score += 0.2

        image_to_show = cv2.imread(f"src/eval/submission/test_images/{class_name}/{str(image_idx)}/image_0.jpg")
        lines = find_lines(image_to_show)

        image_rgb = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)

        # 绘制合并后的直线
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
        for x1, y1, x2, y2 in lines[1]:
            cv2.line(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        plt.clf()
        plt.imshow(image_rgb)
        plt.axis('off')

        plt.savefig(f"src/eval/submission/test_masks/{class_name}/{str(image_idx)}/lines.jpg", dpi=300, bbox_inches='tight')

        horizontal_lines = 0
        vertical_lines = 0

        for line in lines[0]:
            line_length = abs(line[0] - line[2])
            if line_length > 350:
                horizontal_lines += 1
        
        for line in lines[1]:
            line_length = abs(line[1] - line[3])
            if line_length > 350:
                vertical_lines += 1

        if horizontal_lines != 4 or vertical_lines != 6:
            print("line count")
            logical_score += 1
        else:
            pins_centroids = []
            for i in range(len(logical_features)):
                pins_centroids.append(logical_features[i]['centroid'])

            result, grid_counts, _ = check_pin_distribution(pins_centroids, lines[0], lines[1])
            # print(result, grid_counts)
            if not result:
                print("pins distribution")
                logical_score += 1

    if class_name == "splicing_connectors":

        stand_yellow_wire_color = [73, 177, 214.5]
        stand_blue_wire_color = [156, 100.5, 89]
        stand_red_wire_color = [38.5, 62, 184.5]
        

        if abs(logical_features[1]['avg_color'][0] - logical_features[2]['avg_color'][0]) + abs(logical_features[1]['avg_color'][1] - logical_features[2]['avg_color'][1]) + abs(logical_features[1]['avg_color'][2] - logical_features[2]['avg_color'][2]) > 30:
            logical_score += 1
        

        rel_y_1 = get_relative_connection_y(masks[1], masks[0])
        rel_y_2 = get_relative_connection_y(masks[2], masks[0])
        
        area_1 = logical_features[1]['area']
        area_2 = logical_features[2]['area']


        if area_1 >= 5000 and area_2 >= 5000 or (area_1 >= 3000 and area_1 < 5000 and area_2 >= 3000 and area_2 < 5000) or (area_1 < 3000 and area_2 < 3000):
            pass
        else:
            print("area not same")
            logical_score += 1

        if logical_features[0]['area'] < 700:
            print("wire area")
            logical_score += 1


        if rel_y_1 == None or rel_y_2 == None:
            print("no connection")
            logical_score += 1
        else:
            labels = []
            if area_1 >= 2000 and area_1 < 3000:
                if abs(logical_features[0]['avg_color'][0] - stand_yellow_wire_color[0]) > 5 or abs(logical_features[0]['avg_color'][1] - stand_yellow_wire_color[1]) > 11 or abs(logical_features[0]['avg_color'][2] - stand_yellow_wire_color[2]) > 11.5:
                    print("color")
                    logical_score += 1
                for rel_y in [rel_y_1, rel_y_2]:
                    if rel_y >= 0.5:
                        labels.append(1)
                    else:
                        labels.append(0)
            elif area_1 >= 3000 and area_1 < 5000:
                if abs(logical_features[0]['avg_color'][0] - stand_blue_wire_color[0]) > 8 or abs(logical_features[0]['avg_color'][1] - stand_blue_wire_color[1]) > 5.5 or abs(logical_features[0]['avg_color'][2] - stand_blue_wire_color[2]) > 5:
                    print("color")
                    logical_score += 1
                for rel_y in [rel_y_1, rel_y_2]:
                    if rel_y <= 1 / 3:
                        labels.append(0)
                    elif rel_y <= 2 / 3:
                        labels.append(1)
                    else:
                        labels.append(2)
            else:
                rel_y_1 = get_relative_connection_y_by_rotating_mask(masks[1], masks[0])
                rel_y_2 = get_relative_connection_y_by_rotating_mask(masks[2], masks[0])
                if abs(logical_features[0]['avg_color'][0] - stand_red_wire_color[0]) > 2.5 or abs(logical_features[0]['avg_color'][1] - stand_red_wire_color[1]) > 3 or abs(logical_features[0]['avg_color'][2] - stand_red_wire_color[2]) > 9.5:
                    print("color")
                    logical_score += 1
                for rel_y in [rel_y_1, rel_y_2]:
                    if rel_y <= 1 / 5:
                        labels.append(0)
                    elif rel_y <= 2 / 5:
                        labels.append(1)
                    elif rel_y <= 3 / 5:
                        labels.append(2)
                    elif rel_y <= 4 / 5:
                        labels.append(3)
                    else:
                        labels.append(4)
            if labels[0] != labels[1]:
                print("wrong connection")
                logical_score += 1

    if class_name == "screw_bag":
        pass
        if logical_features[4]["enclosing_circle_diameter"] > 200 and logical_features[5]["enclosing_circle_diameter"] > 200:
            logical_score += 1
        if logical_features[4]["enclosing_circle_diameter"] < 200 and logical_features[5]["enclosing_circle_diameter"] < 200:
            logical_score += 1
        if max(logical_features[4]["enclosing_circle_diameter"], logical_features[5]["enclosing_circle_diameter"]) < 200 or min(logical_features[4]["enclosing_circle_diameter"], logical_features[5]["enclosing_circle_diameter"]) < 160:
            logical_score += 1

    return logical_score