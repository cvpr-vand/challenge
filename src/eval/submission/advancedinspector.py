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

                components = self._segment_components(instance_image, instance_mask)
                
                # 收集头部信息
                if components['head_contour'] is not None:
                    all_head_contours.append(components['head_contour'])

        # 创建黄金模板
        if all_head_contours:
            self.golden_head_contour = all_head_contours[0] # 简化处理，用第一个作为模板

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

            components = self._segment_components(instance_image, instance_mask)
            scores = {}

            hsv = cv2.cvtColor(instance_image, cv2.COLOR_BGR2HSV)
            # --- 计算每种异常的分数 ---

            # 2. 污染异常分数
            # cont_mask = cv2.bitwise_or(cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER1, self.CONTAMINATION_RED_UPPER1), cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER2, self.CONTAMINATION_RED_UPPER2))
            cont_mask = cv2.inRange(hsv, self.CONTAMINATION_RED_LOWER1, self.CONTAMINATION_RED_UPPER1)
            scores['contamination_score'] = cv2.countNonZero(cv2.bitwise_and(cont_mask, instance_mask))

            # 3. 头部破损分数
            if components['head_contour'] is not None:
                scores['head_shape_score'] = cv2.matchShapes(self.golden_head_contour, components['head_contour'], cv2.CONTOURS_MATCH_I1, 0.0)
            else:
                 scores['head_shape_score'] = 3.5 # 找不到头，形状异常分数给最大

            predictions.append(scores)
        # --- 【核心修改】聚合所有实例的分数 ---
        if not predictions:
            # 如果图中没有实例，返回一个代表“异常”的分数字典
            return {
                'contamination_score': 30.0, 'head_shape_score': 30.0
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