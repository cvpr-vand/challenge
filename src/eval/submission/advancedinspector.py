import cv2
import numpy as np
from skimage.morphology import skeletonize

class AdvancedInspector:
    """
    V5.0: 最终版、基于物理测量的图钉高级检测器。
    - 遵循“对齐->分割->组件分析”的专业流程。
    - 使用统计和模板匹配方法为每种异常计算分数。
    - 专为Few-shot工作流设计，包含setup和predict方法。
    """
    def __init__(self):
        # --- 可配置参数 ---
        self.HEAD_YELLOW_LOWER = np.array([15, 180, 46])
        self.HEAD_YELLOW_UPPER = np.array([35, 255, 255])
        self.PIN_METAL_LOWER = np.array([0, 0, 50])
        self.PIN_METAL_UPPER = np.array([180, 120, 220])
        self.CONTAMINATION_RED_LOWER1 = np.array([0, 100, 100])
        self.CONTAMINATION_RED_UPPER1 = np.array([10, 255, 255])
        
        # 污染检测的像素数量阈值
        self.CONTAMINATION_PIXEL_THRESHOLD = 32
        # --- Few-shot阶段学习到的“黄金标准”模板 ---
        self.golden_head_contour = None
        self.golden_area = None # 头部轮廓的面积

    def _segment_components(self, aligned_image, aligned_mask):
        hsv_pushpin_only = cv2.bitwise_and(aligned_image, aligned_image, mask=aligned_mask)
        hsv = cv2.cvtColor(hsv_pushpin_only, cv2.COLOR_BGR2HSV)
        head_mask = cv2.inRange(hsv, self.HEAD_YELLOW_LOWER, self.HEAD_YELLOW_UPPER)
        pin_mask = cv2.inRange(hsv, self.PIN_METAL_LOWER, self.PIN_METAL_UPPER)
        kernel = np.ones((3,3), np.uint8)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)
        pin_mask = cv2.morphologyEx(pin_mask, cv2.MORPH_OPEN, kernel)
        head_contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pin_contours, _ = cv2.findContours(pin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return {
            'head_mask': head_mask, 'pin_mask': pin_mask,
            'head_contour': max(head_contours, key=cv2.contourArea) if head_contours else None,
            'pin_contour': max(pin_contours, key=cv2.contourArea) if pin_contours else None
        }
    
    def _find_most_representative_contour(self, contours_list: list):
        """
        [私有方法] 从一组轮廓中找到最具代表性的一个。
        代表性定义为：与列表中其他所有轮廓的平均形状差异最小。
        """
        if not contours_list:
            return None
        if len(contours_list) == 1:
            area = cv2.contourArea(contours_list[0])
            return contours_list[0], area

        min_avg_distance = float('inf')
        best_contour = None
        areas = []
        for i, current_contour in enumerate(contours_list):
            area = cv2.contourArea(current_contour)
            areas.append(area)
            total_distance_to_others = 0
            num_comparisons = 0
            for j, other_contour in enumerate(contours_list):
                if i == j:
                    continue
                
                # 使用cv2.matchShapes计算形状差异
                # CONTOURS_MATCH_I1, I2, I3 都可以尝试，I1通常效果不错
                dist = cv2.matchShapes(current_contour, other_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                total_distance_to_others += dist
                num_comparisons += 1
            
            if num_comparisons == 0: # 应该不会发生，除非列表只有一个元素（已处理）
                avg_distance = 0
            else:
                avg_distance = total_distance_to_others / num_comparisons
            
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_contour = current_contour
        mean_area = np.mean(areas)
        return best_contour, mean_area

    # --- 核心工作流方法 ---
    def setup(self, few_shot_data: list):
        """
        使用Few-shot正常样本来学习和建立“黄金标准”模板。
        
        Args:
            few_shot_data (list): 一个列表，每个元素是一个元组 (image_rgb, foreground_mask)。
        """
        # print("开始Few-shot设置，学习正常样本模板...")
        all_head_contours = []

        for image_rgb, foreground_mask in few_shot_data:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, 8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 100: continue
                x, y, w, h, _ = stats[i]
                instance_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
                instance_image = image_bgr[y:y+h, x:x+w]

                hsv_instance_only = cv2.bitwise_and(instance_image, instance_image, mask=instance_mask)
                hsv_instance = cv2.cvtColor(hsv_instance_only, cv2.COLOR_BGR2HSV)
                h_channel, s_channel, v_channel = cv2.split(hsv_instance)
                # 只检查掩码区域内的 S 和 V 值（非零区域）
                mask_indices = instance_mask > 0
                s_values = s_channel[mask_indices]
                v_values = v_channel[mask_indices]
                # 3. 判断：大多数（比如 85%） S 和 V 都 ≤ 50
                s_thresh_ratio = np.sum(s_values <= 50) / len(s_values)
                v_thresh_ratio = np.sum(v_values <= 50) / len(v_values)
                # 判断 S 和 V 中是否全部 <= 50
                if s_thresh_ratio > 0.85 and v_thresh_ratio > 0.85:
                    print("该区域饱和度和亮度都很低，可能为灰暗区域或背景")
                    continue

                components = self._segment_components(instance_image, instance_mask)
                
                # 收集头部信息
                if components['head_contour'] is not None:
                    all_head_contours.append(components['head_contour'])

        # 创建黄金模板
        if all_head_contours:
            self.golden_head_contour = all_head_contours[0] # 简化处理，用第一个作为模板
            _, self.golden_area = self._find_most_representative_contour(all_head_contours)

    def predict(self, image_rgb: np.ndarray, foreground_mask: np.ndarray):
        """
        对一张新的场景图像进行异常检测，返回每个实例的详细分数。
        """

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, 8)
        
        predictions = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 100: continue
            
            x, y, w, h, _ = stats[i]
            instance_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            instance_image = image_bgr[y:y+h, x:x+w]

            hsv_instance_only = cv2.bitwise_and(instance_image, instance_image, mask=instance_mask)
            hsv_instance = cv2.cvtColor(hsv_instance_only, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv_instance)
            # 只检查掩码区域内的 S 和 V 值（非零区域）
            mask_indices = instance_mask > 0
            s_values = s_channel[mask_indices]
            v_values = v_channel[mask_indices]
            # 3. 判断：大多数（比如 85%） S 和 V 都 ≤ 50
            s_thresh_ratio = np.sum(s_values <= 50) / len(s_values)
            v_thresh_ratio = np.sum(v_values <= 50) / len(v_values)
            # 判断 S 和 V 中是否全部 <= 50
            if s_thresh_ratio > 0.85 and v_thresh_ratio > 0.85:
                print("该区域饱和度和亮度都很低，可能为灰暗区域或背景")
                continue

            components = self._segment_components(instance_image, instance_mask)
            scores = {}

            # hsv = cv2.cvtColor(instance_image, cv2.COLOR_BGR2HSV)
            hsv_pushpin_only = cv2.bitwise_and(instance_image, instance_image, mask=instance_mask)
            hsv = cv2.cvtColor(hsv_pushpin_only, cv2.COLOR_BGR2HSV)
            # --- 计算每种异常的分数 ---

            # 2. 污染异常分数
            # cont_mask = cv2.bitwise_or(cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER1, self.CONTAMINATION_RED_UPPER1), cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER2, self.CONTAMINATION_RED_UPPER2))
            cont_mask = cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER1, self.CONTAMINATION_RED_UPPER1)
            scores['contamination_score'] = cv2.countNonZero(cv2.bitwise_and(cont_mask, instance_mask))

            # 3. 头部破损分数
            if components['head_contour'] is not None:
                scores['head_shape_score'] = cv2.matchShapes(self.golden_head_contour, components['head_contour'], cv2.CONTOURS_MATCH_I1, 0.0)
            else:
                 scores['head_shape_score'] = 10 # 找不到头，形状异常分数给最大

            # 头部面积分数
            if components['head_contour'] is not None:
                head_area = cv2.contourArea(components['head_contour'])
                score = abs(head_area - self.golden_area) / (self.golden_area + 1e-6) # 避免除0
                scores['head_area_score'] = score
            else:
                 scores['head_area_score'] = 10.0

            predictions.append(scores)
        # --- 【核心修改】聚合所有实例的分数 ---
        if not predictions:
            # 如果图中没有实例，返回一个代表“异常”的分数字典
            return {
                'contamination_score': 30.0, 'head_shape_score': 30.0, 'head_area_score': 30.0
            }

        # 初始化图片级别的分数
        image_level_scores = {}
        score_names = predictions[0].keys()

        for name in score_names:
            # 提取所有实例在该维度上的分数
            all_instance_scores = [p.get(name, 0) for p in predictions]
            # 取最大值作为这张图片的最终分数
            image_level_scores[name] = max(all_instance_scores)

        return image_level_scores