import cv2 
import numpy as np
from prompt_ensemble import encode_text_with_prompt_ensemble, encode_normal_text, encode_abnormal_text, encode_general_text, encode_obj_text

def fill_holes(binary_img):
    """
    填充二值图中的空洞
    参数:
        binary_img: 输入的二值图像，uint8类型，前景为255，背景为0
    返回:
        filled_img: 空洞填充后的图像
    """
    # 拷贝图像作为掩膜使用（floodFill需要比原图大2）
    h, w = binary_img.shape
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 复制原图，用于填充操作
    im_floodfill = binary_img.copy()

    # 在边缘某点进行flood fill（假设边缘是背景）
    cv2.floodFill(im_floodfill, floodfill_mask, (0, 0), 255)

    # 对填充结果取反，得到原图中的空洞
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 原图 + 空洞 = 填充空洞后的图
    filled_img = binary_img | im_floodfill_inv

    return filled_img

def detect_slot_anomalies_distance(binary_clamps_image, binary_cable_image, image_width):
    contours, _ = cv2.findContours(binary_clamps_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_corners = []
    right_corners = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        cx = np.mean(box[:, 0])
        
        if cx < image_width / 2:
            top_right = max(box, key=lambda p: p[0] - p[1])
            left_corners.append((tuple(top_right), area))
        else:
            top_left = min(box, key=lambda p: p[0] + p[1])
            right_corners.append((tuple(top_left), area))

    if not left_corners or not right_corners:
        return -1, -1, -1

    left_corners.sort(key=lambda x: x[1], reverse=True)
    right_corners.sort(key=lambda x: x[1], reverse=True)
    left_corner = left_corners[0][0]
    right_corner = right_corners[0][0]

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary_cable_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1, -1, -1

    cable_contour = max(contours, key=cv2.contourArea)
    cable_points = cable_contour[:, 0, :]

    left_end_idx = np.argmin(cable_points[:, 0])
    right_end_idx = np.argmax(cable_points[:, 0])
    left_end = tuple(cable_points[left_end_idx])
    right_end = tuple(cable_points[right_end_idx])

    if abs(left_end[0] - left_corner[0]) > image_width / 4 or abs(right_end[0] - right_corner[0]) > image_width / 4:
        return -1, -1, -1

    left_y_distance = abs(left_end[1] - left_corner[1])
    right_y_distance = abs(right_end[1] - right_corner[1])

    if right_y_distance == 0:
        return -1, left_y_distance, right_y_distance

    distance_ratio = left_y_distance / right_y_distance

    return distance_ratio, left_y_distance, right_y_distance


def Tool_for_screw_bag(self, raw_image):
    img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    color =cv2.resize(img_bgr, self.screw_bag_cicrle_image_shape, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if self.few_shot_inited and self.anomaly_flag is False:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=150, param2=18, minRadius=self.screw_bag_circle_radius[0], maxRadius=self.screw_bag_circle_radius[1])
        if circles is not None :
            circles = np.uint16(np.around(circles))
            if(circles.shape[1] != self.screw_bag_circle_count):
                self.anomaly_flag = True
    
    thresh = cv2.inRange(gray, 19, 255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    selected_mask = np.zeros_like(thresh, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area_index = np.argmax(areas) + 1
    selected_mask[labels == max_area_index] = 255 
    contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:    
        Area_s = cv2.countNonZero(selected_mask) 
        filled_mask = selected_mask.copy()
        contours, hierarchy = cv2.findContours(filled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1 and hierarchy[0][i][2] == -1:
                cv2.drawContours(filled_mask, [cnt], 0, 255, -1)   
        Area_f = cv2.countNonZero(filled_mask) 
        if self.few_shot_inited and self.anomaly_flag is False:
            if (Area_f - Area_s) > 150:
                self.anomaly_flag = True
            
        contours_filled = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        max_contour_filled = max(contours_filled, key=cv2.contourArea)
        Area_c = cv2.contourArea(max_contour_filled)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        contours_closed = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        max_contour_closed = max(contours_closed, key=cv2.contourArea)
        Area_ss = cv2.contourArea(max_contour_closed)
        if not self.few_shot_inited:
            area_difference_ss_c = Area_ss - Area_c
        else:
            area_difference_ss_c = None
        if self.few_shot_inited and self.area_difference_ss_c_save != 0 and self.anomaly_flag is False:
            ratio2 = (Area_ss - Area_c) / self.area_difference_ss_c_save
            if ratio2 > 1.5:
                self.anomaly_flag = True
    return self.anomaly_flag, area_difference_ss_c


def Tool_for_splicing_connectors(self, raw_image, sam_mask_max_area, patch_merge_sam, instance_masks, proj_patch_token, test_mode):
    def detect_color(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        blue_lower = np.array([90, 100, 100])
        blue_upper = np.array([120, 255, 255])
        
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        max_count = max(np.sum(red_mask), np.sum(yellow_mask), np.sum(blue_mask))
        
        if max_count == np.sum(red_mask):
            return "red"
        elif max_count == np.sum(yellow_mask):
            return "yellow"
        else:
            return "blue"
        
    
    origin_image = cv2.cvtColor(raw_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask = sam_mask_max_area * 255
    _, binary = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY_INV)
    height, width = binary.shape[:2] 
    left=int(width*0.5-30)
    right=int(width*0.5+30)
    middle_region = binary[:, left:right]
    contours, _ = cv2.findContours(middle_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = ""
    if len(contours) == 1:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        x += left
        middle_region_org = origin_image[y:y+h, x:x+w]
        color = detect_color(middle_region_org)
    else:
        self.anomaly_flag = True

    if color=="blue":
        B, G, R = cv2.split(origin_image)
        _, thresholded = cv2.threshold(R, 245, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        filled = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        total_area = cv2.countNonZero(filled)
        if total_area < 5100:
           self.anomaly_flag = True
               
    if color=="red":
        B, G, R = cv2.split(origin_image)
        _, thresholded = cv2.threshold(R, 245, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        filled = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        total_area = cv2.countNonZero(filled)
        if total_area < 8650:
           self.anomaly_flag = True    
  
    if color=="yellow":
        B, G, R = cv2.split(origin_image)
        _, thresholded = cv2.threshold(R, 220, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        filled = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        total_area = cv2.countNonZero(filled)
        if total_area < 6080:
           self.anomaly_flag = True
 
    if test_mode and self.anomaly_flag:
        binary_connector = binary_clamps = binary_cable = np.zeros((self.feat_size, self.feat_size), dtype=np.uint8)
        distance = distance_ratio = foreground_pixel_count = 1
        return instance_masks,binary_connector,binary_clamps,binary_cable, distance, distance_ratio, foreground_pixel_count
           
    def get_bounding_box_length(contour):
        rect = cv2.minAreaRect(contour)
        (width, height) = rect[1]
        return max(width, height)

    def get_bounding_box_width(contour):
        rect = cv2.minAreaRect(contour)
        (width, height) = rect[1]
        return min(width, height)

    def get_angle_from_contour(contour):
        if len(contour) < 2:
            return 0
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return abs(angle)

    def process_regions(mask):
        kernel = np.ones((3, 3), np.uint8)
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 500:
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                regions.append({'contour': cnt, 'area': area, 'center': (cX, cY)})
        return regions

    def count_large_holes(mask, min_hole_area=30):
        mask_copy = mask.copy()
        contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return False
        for i, contour_info in enumerate(hierarchy[0]):
            parent_id = contour_info[3]
            if parent_id != -1:
                hole_area = cv2.contourArea(contours[i])
                if hole_area >= min_hole_area:
                    return True
        return False

    def get_region_thickness(mask, edge_margin=20):
        h, w = mask.shape
        if cv2.countNonZero(mask) < 500:
            return None, None
        mean, eigenvectors = cv2.PCACompute(mask.astype(np.float32), mean=np.array([]))
        main_dir = eigenvectors[0]
        angle = np.degrees(np.arctan2(main_dir[1], main_dir[0]))

        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)
        rotated_mask = cv2.warpAffine(mask, rot_mat, (w, h))

        coords = cv2.findNonZero(rotated_mask)
        if coords is None:
            return None, None

        x, y, width, height = cv2.boundingRect(coords)

        thickness_list = []
        for i in range(x + edge_margin, x + width - edge_margin):
            col = rotated_mask[:, i]
            indices = np.where(col > 0)[0]
            if len(indices) >= 2:
                thickness = indices[-1] - indices[0]
                thickness_list.append(thickness)

        if not thickness_list:
            return None, None

        max_t = max(thickness_list)
        min_t = min(thickness_list)
        return max_t, min_t

    def detect_image(image):
        results = {}

        image_resized = cv2.resize(image, (896, 448), interpolation=cv2.INTER_CUBIC)
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

        saturation = hsv_image[:, :, 1]
        mask_saturation = cv2.inRange(saturation, 75, 255)
        mask_new_range = cv2.inRange(hsv_image, np.array([10, 0, 194]), np.array([38, 12, 255]))
        new_range_regions = process_regions(mask_new_range)

        mask_red = cv2.inRange(hsv_image, np.array([0, 200, 0]), np.array([255, 255, 255]))
        mask_blue = cv2.inRange(hsv_image, np.array([80, 90, 0]), np.array([150, 255, 255]))
        mask_yellow = cv2.inRange(hsv_image, np.array([12, 115, 0]), np.array([80, 255, 255]))
        mask_green = cv2.inRange(hsv_image, np.array([50, 20, 0]), np.array([98, 255, 255]))

        RedRegions = process_regions(mask_red)
        BlueRegions = process_regions(mask_blue)
        YellowRegions = process_regions(mask_yellow)
        saturationRegions = process_regions(mask_saturation)
        GreenRegions = process_regions(mask_green)

        AreaRed = [r['area'] for r in RedRegions]
        NumberRed = len(RedRegions)
        AreaBlue = [r['area'] for r in BlueRegions]
        NumberBlue = len(BlueRegions)
        AreaYellow = [r['area'] for r in YellowRegions]
        NumberYellow = len(YellowRegions)
        NumberGreen = len(GreenRegions)

        LengthlineR = [get_bounding_box_length(r['contour']) for r in RedRegions]
        LengthlineB = [get_bounding_box_length(r['contour']) for r in BlueRegions]
        LengthlineY = [get_bounding_box_length(r['contour']) for r in YellowRegions]
        WidthLineR = [get_bounding_box_width(r['contour']) for r in RedRegions]
        WidthLineB = [get_bounding_box_width(r['contour']) for r in BlueRegions]
        WidthLineY = [get_bounding_box_width(r['contour']) for r in YellowRegions]
        LengthlineS = sum(get_bounding_box_length(r['contour']) for r in saturationRegions)

        rules = {}

        def add_thickness_rule(regions, mask, name):
            if len(regions) == 0:
                return
            largest_contour = max(regions, key=lambda x: x['area'])
            largest_mask = np.zeros_like(mask)
            cv2.drawContours(largest_mask, [largest_contour['contour']], -1, 255, thickness=cv2.FILLED)
            max_t, min_t = get_region_thickness(largest_mask, edge_margin=20)
            if max_t is not None and min_t is not None:
                diff = max_t - min_t
                rule_name = f"{name}区厚度变化<10"
                rules[rule_name] = "NG" if diff > 9 else "OK"

        if NumberBlue >= 1:
            add_thickness_rule(BlueRegions, mask_blue, "蓝")
        if NumberYellow >= 1:
            add_thickness_rule(YellowRegions, mask_yellow, "黄")
        rules['规则A：红色区域≥3'] = 'NG' if NumberRed >= 3 else 'OK'

        rules['规则B：检测到绿色区域'] = 'NG' if NumberGreen >= 1 else 'OK'

        rules['规则C：两红无黄无蓝'] = 'NG' if (NumberRed == 2 and NumberYellow == 0 and NumberBlue == 0) else 'OK'

        has_yellow_holes = count_large_holes(mask_yellow, 20)
        rules['规则D：黄色区域存在大孔洞'] = 'NG' if has_yellow_holes else 'OK'

        has_blue_holes = count_large_holes(mask_blue, 20)
        rules['规则E：蓝色区域存在大孔洞'] = 'NG' if has_blue_holes else 'OK'
        large_saturation_regions = [r for r in saturationRegions if r['area'] > 1000]
        rules['规则F：saturation区域面积>500的数量≥2'] = 'NG' if len(large_saturation_regions) >= 2 else 'OK'
        if any(width > 39.9 for width in WidthLineB):
            rules['规则G：蓝区宽度>39.9'] = 'NG'
        else:
            rules['规则G：蓝区宽度>39.9'] = 'OK'

        if any(width > 39.9 for width in WidthLineY):
            rules['规则H：黄区宽度>39.9'] = 'NG'
        else:
            rules['规则H：黄区宽度>39.9'] = 'OK'
        red_angle_degrees = None

        rules['规则I：红区差集角度>10'] = 'OK'
        if NumberRed == 1:
            red_contour = RedRegions[0]['contour']
            red_mask = np.zeros_like(mask_red)
            cv2.drawContours(red_mask, [red_contour], -1, 255, thickness=cv2.FILLED)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            red_opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open, iterations=10)
            diff = cv2.subtract(red_mask, red_opened)
            diff_contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(diff_contours) > 0:
                largest_contour = max(diff_contours, key=cv2.contourArea)

                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                angle = rect[-1]  
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                angle = abs(angle)

                if angle > 5.5:
                    rules['规则I：红区差集角度>10'] = 'NG'
                else:
                    rules['规则I：红区差集角度>10'] = 'OK'
            else:
                rules['规则I：红区差集角度>10'] = 'OK'
        else:
            rules['规则I：红区差集角度>10'] = 'OK'

        if NumberRed == 2:
            if NumberYellow == 0 and NumberBlue == 0:
                rules['规则1'] = 'NG'
            else:
                rules['规则1'] = 'OK'
        else:
            rules['规则1'] = 'OK'

        NG_flag = False
        for area in AreaRed:
            if 4000 <= area <= 6000:
                if NumberYellow == 0:
                    NG_flag = True
                    break
        rules['规则2'] = 'NG' if NG_flag else 'OK'

        NG_flag = False
        for area in AreaRed:
            if 6500 <= area <= 8000:
                if NumberBlue == 0:
                    NG_flag = True
                    break
        rules['规则3'] = 'NG' if NG_flag else 'OK'

        NG_flag = False
        if NumberRed == 1:
            if not (28300 <= AreaRed[0] <= 32000):
                NG_flag = True
            if NumberYellow > 0 or NumberBlue > 0:
                NG_flag = True
        rules['规则4'] = 'NG' if NG_flag else 'OK'

        if LengthlineS < 500:
            rules['规则5'] = 'NG'
        else:
            rules['规则5'] = 'OK'

        if NumberRed == 2:
            if abs(AreaRed[0] - AreaRed[1]) > 400:
                rules['规则6'] = 'NG'
            else:
                rules['规则6'] = 'OK'
        else:
            rules['规则6'] = 'OK'

        if NumberBlue == 2 or NumberYellow == 2:
            rules['规则7'] = 'NG'
        else:
            rules['规则7'] = 'OK'

        if any(area > 8000 for area in AreaYellow):
            rules['新规则1（Yellow面积>7000）'] = 'NG'
        else:
            rules['新规则1（Yellow面积>7000）'] = 'OK'

        if NumberRed == 2:
            if any(area > 9000 for area in AreaRed):
                rules['新规则2（红两个，其中一个面积>7000）'] = 'NG'
            else:
                rules['新规则2（红两个，其中一个面积>7000）'] = 'OK'
        else:
            rules['新规则2（红两个，其中一个面积>7000）'] = 'OK'

        yellow_position_info = ''
        if NumberRed == 2 and NumberYellow == 1:
            yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_yellow_centers = []
            for y_cnt in yellow_contours:
                y_area = cv2.contourArea(y_cnt)
                if 100 <= y_area <= 500:
                    M_y = cv2.moments(y_cnt)
                    if M_y["m00"] != 0:
                        cy_x = int(M_y["m10"] / M_y["m00"])
                        cy_y = int(M_y["m01"] / M_y["m00"])
                        filtered_yellow_centers.append((cy_x, cy_y))
            if len(filtered_yellow_centers) >= 1:
                red_rects = []
                for region in RedRegions:
                    x, y, w, h = cv2.boundingRect(region['contour'])
                    red_rects.append((x, y, w, h))
                yellow_codes = []
                for rect in red_rects:
                    x, y, w, h = rect
                    rect_center_y = y + h / 2
                    found = False
                    for (cy_x, cy_y) in filtered_yellow_centers:
                        if x <= cy_x <= x + w and y <= cy_y <= y + h:
                            code = 0 if cy_y < rect_center_y else 1
                            yellow_codes.append(code)
                            found = True
                            break
                    if not found:
                        yellow_codes.append(-1)
                if len(yellow_codes) == 2 and yellow_codes[0] in (0, 1) and yellow_codes[1] in (0, 1):
                    yellow_position_info = f"({yellow_codes[0]},{yellow_codes[1]})"
        rules['黄区标记位置'] = yellow_position_info

        if yellow_position_info and len(yellow_position_info) >= 5:
            a = int(yellow_position_info[1])
            b = int(yellow_position_info[3])
            if a != b:
                rules['黄区标记位置不一致'] = 'NG'
            else:
                rules['黄区标记位置不一致'] = 'OK'
        else:
            rules['黄区标记位置不一致'] = 'OK'

        blue_position_info = ''
        if NumberRed == 2 and NumberBlue == 1:
            mask_blue_new = cv2.inRange(hsv_image, np.array([75, 37, 0]), np.array([150, 255, 255]))
            blue_contours, _ = cv2.findContours(mask_blue_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_blue_centers = []
            for b_cnt in blue_contours:
                b_area = cv2.contourArea(b_cnt)
                if 100 <= b_area <= 500:
                    M_b = cv2.moments(b_cnt)
                    if M_b["m00"] != 0:
                        bx = int(M_b["m10"] / M_b["m00"])
                        by = int(M_b["m01"] / M_b["m00"])
                        filtered_blue_centers.append((bx, by))
            if len(filtered_blue_centers) >= 1:
                red_rects = []
                for region in RedRegions:
                    x, y, w, h = cv2.boundingRect(region['contour'])
                    red_rects.append((x, y, w, h))
                blue_codes = []
                for rect in red_rects:
                    x, y, w, h = rect
                    step = h / 3
                    for (bx, by) in filtered_blue_centers:
                        if x <= bx <= x + w and y <= by <= y + h:
                            if by < y + step:
                                code = 0
                            elif y + step <= by < y + 2 * step:
                                code = 1
                            else:
                                code = 2
                            blue_codes.append(code)
                            break
                    else:
                        blue_codes.append(-1)
                if len(blue_codes) == 2 and blue_codes[0] in (0, 1, 2) and blue_codes[1] in (0, 1, 2):
                    blue_position_info = f"({blue_codes[0]},{blue_codes[1]})"
        rules['蓝区标记位置'] = blue_position_info
        has_new_range = any(region['area'] > 20 for region in new_range_regions)
        rules['新增规则4：HSV特定范围区域存在面积>20'] = 'NG' if has_new_range else 'OK'

        if blue_position_info and len(blue_position_info) >= 5:
            a = int(blue_position_info[1])
            b = int(blue_position_info[3])
            if a != b:
                rules['蓝区标记位置不一致'] = 'NG'
            else:
                rules['蓝区标记位置不一致'] = 'OK'
        else:
            rules['蓝区标记位置不一致'] = 'OK'

        rules['Overall'] = 'NG' if any(v == 'NG' for v in rules.values()) else 'OK'
        return rules

    rules = detect_image(origin_image)
    if rules['Overall'] == 'NG':
        self.anomaly_flag = True

    if test_mode and self.anomaly_flag:
        binary_connector = binary_clamps = binary_cable = np.zeros((self.feat_size, self.feat_size), dtype=np.uint8)
        distance = distance_ratio = foreground_pixel_count = 1
        return instance_masks,binary_connector,binary_clamps,binary_cable, distance, distance_ratio, foreground_pixel_count


    binary = (sam_mask_max_area == 0).astype(np.uint8) 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = 0
    for i in range(1, num_labels):
        temp_mask = labels == i
        if np.sum(temp_mask) <= 64: 
            binary[temp_mask] = 0 
        else:
            count += 1
    if count != 1 and self.anomaly_flag is False: 
        self.anomaly_flag = True

    patch_merge_sam[~(binary.astype(np.bool_))] = self.patch_query_obj.shape[0] - 1 

    kernel = np.ones((23, 23), dtype=np.uint8)
    erode_binary = cv2.erode(binary, kernel)
    h, w = erode_binary.shape
    distance = 0

    left, right = erode_binary[:, :int(w/2)],  erode_binary[:, int(w/2):]   
    left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
    right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1] # foreground

    binary_cable = (patch_merge_sam == 1).astype(np.uint8)
    binary_cable_orinal = binary_cable.copy()
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary_cable = cv2.erode(binary_cable, kernel)
    kernel1 = np.ones((5, 5),  dtype=np.uint8)
    binary_cable1 = cv2.erode(binary_cable, kernel1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cable, connectivity=8)
    for i in range(1, num_labels):
        temp_mask = labels == i
        if np.sum(temp_mask) <= 64: # 448x448
            binary_cable[temp_mask] = 0 # set to background

    binary_cable = cv2.resize(binary_cable, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
    
    binary_clamps = (patch_merge_sam == 0).astype(np.uint8)
    binary_clamps_orinal = binary_clamps.copy()

    kernel = np.ones((5, 5), dtype=np.uint8)
    binary_clamps = cv2.erode(binary_clamps, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clamps, connectivity=8)
    for i in range(1, num_labels):
        temp_mask = labels == i
        if np.sum(temp_mask) <= 64: # 448x448
            binary_clamps[temp_mask] = 0 # set to background
        else:
            instance_mask = temp_mask.astype(np.uint8)
            instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            if instance_mask.any():
                instance_masks.append(instance_mask.astype(np.bool_).reshape(-1))

    binary_clamps = cv2.resize(binary_clamps, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

    binary_connector = cv2.resize(binary, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
    
    query_cable_color = encode_obj_text(self.model_clip, self.splicing_connectors_cable_color_query_words_dict, self.tokenizer, self.device)
    cable_feature = proj_patch_token[binary_cable.astype(np.bool_).reshape(-1), :].mean(0, keepdim=True)
    idx_color = (cable_feature @ query_cable_color.T).argmax(-1).squeeze(0).item()
    foreground_pixel_count = np.sum(erode_binary) / self.splicing_connectors_count[idx_color]

    slice_cable = binary_cable1[:, int(w/2)-5: int(w/2)+5]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
    cable_count = num_labels - 1
    if cable_count != 1 and self.anomaly_flag is False:
        self.anomaly_flag = True

    if self.few_shot_inited and self.foreground_pixel_hist != 0 and self.anomaly_flag is False:
        ratio = foreground_pixel_count / self.foreground_pixel_hist
        if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:
            self.anomaly_flag = True
            print('cable color and number of clamps mismatch, cable color idx: {} (0: yellow 2-clamp, 1: blue 3-clamp, 2: red 5-clamp), foreground_pixel_count :{}, canonical foreground_pixel_hist: {}.'.format(idx_color, foreground_pixel_count, self.foreground_pixel_hist))


    ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
    if self.few_shot_inited and (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False: 
        self.anomaly_flag = True
        print('left and right connectors are not symmetry.left_count is {}, right_count is {}'.format(np.sum(left_count) , np.sum(right_count)))
        

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode_binary, connectivity=8)
    if num_labels - 1 == 2:
        centroids = centroids[1:]
        x1, y1 = centroids[0] 
        x2, y2 = centroids[1]
        distance = np.sqrt((x1/w - x2/w)**2 + (y1/h - y2/h)**2)
        if self.few_shot_inited and self.splicing_connectors_distance != 0 and self.anomaly_flag is False:
            ratio = distance / self.splicing_connectors_distance
            if ratio < 0.6 or ratio > 1.4:
                print('cable is too short or too long. ratio is {},distance is {}, self.splicing_connectors_distaces is {}'.format(ratio, distance, self.splicing_connectors_distance))
                self.anomaly_flag = True


    if not self.few_shot_inited:
        distance_ratio, _, _= detect_slot_anomalies_distance(binary_clamps_orinal, binary_cable_orinal, w)
    else:
        distance_ratio = None 



    if self.few_shot_inited and self.distance_ratio_save != 0 and  self.anomaly_flag is False:
        distance_ratio, left_y_distance, right_y_distance = detect_slot_anomalies_distance(binary_clamps_orinal, binary_cable_orinal, w)
        if distance_ratio != -1:
            ratio = distance_ratio / self.distance_ratio_save
            if 0.1 < distance_ratio < 0.4 or distance_ratio > 1.28:
                self.anomaly_flag = True
                print('two side is not match. ratio is {}, distance_ratio is {}, self.distance_ratio_save is {}'.format(ratio, distance_ratio, self.distance_ratio_save))
                print('left_y_distance is {} ,right_y_distance is {}'.format(left_y_distance, right_y_distance))

    return instance_masks,binary_connector,binary_clamps,binary_cable, distance, distance_ratio, foreground_pixel_count


def Tool_for_pushpins(self, img, angle_thresh=15):
    result = {
        "Overall": "OK",
        "Yellow_Count": "OK",
        "Yellow_Area": "OK",
        "Red_Area": "OK",
        "Max_Edge": "OK",
        "Convex_Hull": "OK",
        "Yellow_in_Hull": "OK",
        "Diff_Region_Count": "OK",
        "Diff_Region_Num": 0,
        "Hull_Direction": "OK"
    }
    ng_flag = False
    max_edge_list = []
    hull_direction_ok = True

    resized_img = cv2.resize(img, (762, 448))
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    kernel = np.ones((3, 3), np.uint8)

    yellow_lower1 = np.array([15, 140, 0])
    yellow_upper1 = np.array([240, 240, 255])
    yellow_lower2 = np.array([35, 100, 100])
    yellow_upper2 = np.array([35, 255, 255])

    mask_yellow1 = cv2.inRange(hsv, yellow_lower1, yellow_upper1)
    mask_yellow2 = cv2.inRange(hsv, yellow_lower2, yellow_upper2)
    yellow_mask = cv2.bitwise_or(mask_yellow1, mask_yellow2)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_yellow_contours = [c for c in yellow_contours if cv2.contourArea(c) > 30]
    yellow_count = len(large_yellow_contours)
    result["Yellow_Count"] = "OK" if yellow_count == 15 else "NG"
    if result["Yellow_Count"] == "NG":
        ng_flag = True

    for c in large_yellow_contours:
        area = cv2.contourArea(c)
        if area > 1500 or area < 890:
            result["Yellow_Area"] = "NG"
            ng_flag = True

    red_lower1 = np.array([0, 210, 0])
    red_upper1 = np.array([14, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_red_area = sum([cv2.contourArea(c) for c in red_contours])
    result["Red_Area"] = "OK" if total_red_area <= 12 else "NG"
    if result["Red_Area"] == "NG":
        ng_flag = True

    b_channel = resized_img[:,:,0]
    b_mask = cv2.inRange(b_channel, 0, 52)
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)

    contours_b, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_filtered_count = 0
    has_empty_hull = False  

    diff_region_count = 0

    for c in contours_b:
        if len(c) >= 3:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if 10000 <= hull_area <= 99999:
                area_filtered_count += 1

                hull_mask = np.zeros_like(b_mask)
                cv2.fillPoly(hull_mask, [hull], 255)
                yellow_in_hull = cv2.bitwise_and(yellow_mask, hull_mask)
                y_cnts_n, _ = cv2.findContours(yellow_in_hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                angle_y_in_hull = None
                yellow_area_in_hull = 0
                for c_y in y_cnts_n:
                    if cv2.contourArea(c_y) > 10:
                        rect_y = cv2.minAreaRect(c_y)
                        angle_y_in_hull = rect_y[2]
                        yellow_area_in_hull += cv2.contourArea(c_y)
                        break

                if yellow_area_in_hull == 0:
                    has_empty_hull = True
                    angle_y_in_hull = None

                mask_poly = np.zeros_like(b_mask)
                cv2.fillPoly(mask_poly, [hull], 255)
                shrink_pixels = 5
                inner_mask = cv2.erode(mask_poly, kernel, iterations=shrink_pixels)

                coords = np.column_stack(np.where(inner_mask > 0))
                if coords.shape[0] == 0:
                    continue
                inner_v_values = v_channel[coords[:,0], coords[:,1]]
                if inner_v_values.size == 0:
                    continue

                in70_255_mask = (inner_v_values >= 65) & (inner_v_values <= 255)
                in180_255_mask = (inner_v_values >= 190) & (inner_v_values <= 255)

                full_shape = v_channel.shape
                mask_180_ext = np.zeros(full_shape, dtype=np.uint8)
                mask_70_full = np.zeros(full_shape, dtype=np.uint8)

                for i, (y,x) in enumerate(coords):
                    if in180_255_mask[i]:
                        mask_180_ext[y,x] = 255
                    if in70_255_mask[i]:
                        mask_70_full[y,x] = 255

                ext_mask = cv2.dilate(mask_180_ext, np.ones((8,8),np.uint8), iterations=2)
                diff_mask = cv2.bitwise_and(mask_70_full, cv2.bitwise_not(ext_mask))
                contours_diff, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                angle_diff_in_this_hull = []
                has_direction_ng = False

                for cdiff in contours_diff:
                    if cv2.contourArea(cdiff) > 1:
                        rect2 = cv2.minAreaRect(cdiff)
                        (x0, y0), (w2, h2), angle2 = rect2
                        max_edge = max(w2, h2)
                        if max_edge >= 6:
                            diff_region_count += 1
                            if angle_y_in_hull is not None:
                                ang1 = angle_y_in_hull
                                ang2 = angle2
                                a1 = ang1 if ang1 >= 0 else ang1 + 90
                                a2 = ang2 if ang2 >= 0 else ang2 + 90
                                diff = abs(a1 - a2)
                                if diff > angle_thresh and abs(diff - 90) > angle_thresh:
                                    has_direction_ng = True
                            angle_diff_in_this_hull.append(angle2)
                        max_edge_list.append(max_edge)

                if angle_y_in_hull is not None and has_direction_ng:
                    hull_direction_ok = False 

    result["Convex_Hull"] = "OK" if area_filtered_count == 15 else "NG"
    if area_filtered_count != 15:
        ng_flag = True

    if max_edge_list and min(max_edge_list) < 0: 
        result["Max_Edge"] = "NG"
        ng_flag = True

    result["Yellow_in_Hull"] = "OK"
    if has_empty_hull:
        result["Yellow_in_Hull"] = "NG"
        ng_flag = True

    result["Hull_Direction"] = "OK" if hull_direction_ok else "NG"
    if not hull_direction_ok:
        ng_flag = True

    result["Diff_Region_Num"] = diff_region_count
    result["Diff_Region_Count"] = "OK" if diff_region_count == 15 else "NG"
    if diff_region_count != 15:
        ng_flag = True

    result["Overall"] = "NG" if ng_flag else "OK"
    if ng_flag:
        self.anomaly_flag = True
    return result


def Tool_for_breakfast_box(self, raw_image, patch_merge_sam):

    def apply_watershed(binary):
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown==255] = 0
        color_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color_binary, markers)
        
        contours = []
        for mark in np.unique(markers):
            if mark <= 1: continue
            temp = np.zeros_like(binary, dtype=np.uint8)
            temp[markers == mark] = 255
            cnts, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend([c for c in cnts if cv2.contourArea(c) > 5000])
        
        return contours[:3]
            
    def classify_fruit(contour, original_img):
        mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        x,y,w,h = cv2.boundingRect(contour)
        roi = original_img[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hue_mean = np.mean(hsv[:,:,0][mask_roi == 255])
        orange_mask = cv2.inRange(hsv, (10, 100, 100), (25, 255, 255))
        orange_ratio = np.sum(orange_mask > 0) / np.sum(mask_roi == 255)
        
        b, g, r = cv2.split(roi)
        rg_ratio = np.mean(r[mask_roi == 255]) / np.mean(g[mask_roi == 255])
        
        if hue_mean < 10:
            if (6.8 < hue_mean <= 6.85 
                and orange_ratio >= 0.2 
                and 2.31 < rg_ratio < 2.39):
                return "peach", hue_mean, orange_ratio, rg_ratio
            elif (9.4 < hue_mean < 9.6 
                and orange_ratio >= 0.3 
                and rg_ratio < 2.3):
                return "peach", hue_mean, orange_ratio, rg_ratio
            else:
                return "orange", hue_mean, orange_ratio, rg_ratio
        else:
            return "peach", hue_mean, orange_ratio, rg_ratio

    def validate_counts(results):
        oranges = sum(1 for r in results if r['type'] == 'orange')
        peaches = sum(1 for r in results if r['type'] == 'peach')
        return 0 if (oranges == 2 and peaches == 1) else 1


    origin_image = cv2.cvtColor(raw_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    sam_max_idx = np.max(patch_merge_sam)
    mask = patch_merge_sam * (255 // sam_max_idx - 1)

    h, w = mask.shape
    left_mask = mask[:, :int(w*0.5)]
    binary = cv2.inRange(left_mask, 0, 60)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    
    if len(valid_contours) < 3:
        valid_contours = apply_watershed(binary)
        if len(valid_contours) < 3:
            self.anomaly_flag = True
            return

    results = []
    for cnt in valid_contours:
        fruit_type, hue_mean,orange_ratio,rg_ratio = classify_fruit(cnt, origin_image)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00']) if M['m00'] > 0 else 0
        cy = int(M['m01']/M['m00']) if M['m00'] > 0 else 0
        
        results.append({
            "type": fruit_type,
            "center": (cx, cy),
            "contour": cnt,
            "area": cv2.contourArea(cnt),
            "hue_mean": hue_mean,
            "orange_ratio": orange_ratio,
            "rg_ratio": rg_ratio
        })
    
    results.sort(key=lambda x: x["center"][1])
    status = validate_counts(results)
    if status == 1: 
        self.anomaly_flag = True


def Tool_for_juice_bottle(raw_image, patch_merge_sam):
    raw_image = cv2.cvtColor(raw_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    sam_max_idx = np.max(patch_merge_sam)
    sam_image = patch_merge_sam * (255 // sam_max_idx - 1)
    sam_image = sam_image.astype(np.uint8)

    a = 90000
    b = 135000
    c = 50
    min_area = 100

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    target_size = (896, 448)  # width, height
    results = []

    fruit_offset_ranges = {
        "橙子": {"x": (115, 127), "y": (25, 32)},
        "香蕉": {"x": (163, 180), "y": (35, 43)},
        "樱桃": {"x": (125, 158), "y": (42, 48)}
    }

    hsv_ranges = {
        "橙子": ((0, 108), (0, 255), (73, 255)),
        "香蕉": ((0, 108), (0, 255), (73, 255)),
        "樱桃": ((0, 255), (54, 255), (40, 255))
    }

    def get_largest_region_and_height(hsv_img, hsv_range):
        h_low, h_high = hsv_range[0]
        s_low, s_high = hsv_range[1]
        v_low, v_high = hsv_range[2]

        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return mask, h

    sam_image = cv2.resize(sam_image, target_size, interpolation=cv2.INTER_NEAREST)
    raw_image = cv2.resize(raw_image, target_size, interpolation=cv2.INTER_LINEAR)

    result = {
        "文件名": "",
        "液体区域OK": True,
        "液体长度OK": True,
        "标签纸数量OK": True,
        "标签纸对齐OK": True,
        "标签纸宽度OK": True,
        "水果圆度OK": True,
        "标签纸矩形度OK": True,
        "水果液体匹配OK": True,
        "水果偏移OK": True,
        "水果长度OK": True,
        "标签纸面积1": 0,
        "标签纸面积2": 0,
        "水果类型": "未知",
        "液体类型": "未知",
        "水果偏移_x": np.nan,
        "水果偏移_y": np.nan,
        "橙子_x偏移": np.nan,
        "橙子_y偏移": np.nan,
        "樱桃_x偏移": np.nan,
        "樱桃_y偏移": np.nan,
        "香蕉_x偏移": np.nan,
        "香蕉_y偏移": np.nan,
        "液体区域长度": np.nan,
        "水果区域长度": np.nan,
        "标签纸面积顺序OK": True,
        "标签纸间距OK": True,
        "小标签V通道总面积": 0,
        "小标签V通道面积和NG": False,
        "Overall": "OK"
    }

    reasons = []

    liquid_mask = (sam_image == 62).astype(np.uint8)
    liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not valid_contours:
        result["液体区域OK"] = False
        reasons.append("没有有效液体区域")
        result["液体类型"] = "未识别"
        result["液体区域长度"] = np.nan
    else:
        liquid_contour = max(valid_contours, key=cv2.contourArea)
        liquid_area = cv2.contourArea(liquid_contour)

        if liquid_area < a or liquid_area > b:
            result["液体区域OK"] = False
            reasons.append(f"液体区域面积 {liquid_area} 不在 [{a}, {b}] 范围内")


        liquid_region_mask = np.zeros_like(sam_image)
        cv2.drawContours(liquid_region_mask, [liquid_contour], -1, 255, thickness=cv2.FILLED)

        b_channel = raw_image[:, :, 0]
        mean_b = cv2.mean(b_channel, mask=liquid_region_mask)[0]

        if 38 <= mean_b <= 55:
            result["液体类型"] = "樱桃水"
        elif 64 <= mean_b <= 76:
            result["液体类型"] = "橙子水"
        elif mean_b >= 90:
            result["液体类型"] = "香蕉水"
        else:
            result["液体类型"] = f"其他({mean_b:.1f})"

        x_liquid, y_liquid, w_liquid, h_liquid = cv2.boundingRect(liquid_contour)
        result["液体区域长度"] = h_liquid

        if h_liquid > 340 or h_liquid < 290:
            result["液体长度OK"] = False
            reasons.append(f"液体区域长度 {h_liquid} 不在允许范围内")

    label_mask = (sam_image == 186).astype(np.uint8)
    label_contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_label_contours = [cnt for cnt in label_contours if cv2.contourArea(cnt) > min_area]

    if len(valid_label_contours) != 2:
        result["标签纸数量OK"] = False
        reasons.append(f"标签纸个数 {len(valid_label_contours)} ≠ 2")
    else:
        label_rects = [cv2.boundingRect(cnt) for cnt in valid_label_contours]

        area1 = cv2.contourArea(valid_label_contours[0])
        area2 = cv2.contourArea(valid_label_contours[1])
        result["标签纸面积1"] = area1
        result["标签纸面积2"] = area2

        largest_label_contour = max(valid_label_contours, key=cv2.contourArea)
        x_largest, y_largest, w_largest, h_largest = cv2.boundingRect(largest_label_contour)
        label_top_left = (x_largest, y_largest)

        rect1, rect2 = label_rects
        cnt1, cnt2 = valid_label_contours
        area1 = cv2.contourArea(cnt1)
        area2 = cv2.contourArea(cnt2)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        if area1 > area2:
            large_rect = rect1
            small_rect = rect2
        else:
            large_rect = rect2
            small_rect = rect1

        large_y = large_rect[1]
        small_y = small_rect[1]

        if large_y >= small_y:
            result["标签纸面积顺序OK"] = False
            reasons.append("面积大的标签纸不在上方")

        upper_rect = rect1 if y1 < y2 else rect2
        lower_rect = rect1 if y1 > y2 else rect2

        upper_bottom_y = upper_rect[1] + upper_rect[3]
        lower_top_y = lower_rect[1]
        distance = lower_top_y - upper_bottom_y

        if distance < 45 or distance > 80:
            result["标签纸间距OK"] = False
            reasons.append(f"标签纸间距 {distance} 不符合要求")

        hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

        orange_lower = np.array([32, 80, 0])
        orange_upper = np.array([79, 255, 255])

        cherry_lower = np.array([0, 145, 93])
        cherry_upper = np.array([109, 207, 172])

        label_region_mask = np.zeros_like(sam_image)
        largest_label_contour = max(valid_label_contours, key=cv2.contourArea)
        cv2.drawContours(label_region_mask, [largest_label_contour], -1, 255, thickness=cv2.FILLED)

        orange_mask = cv2.bitwise_and(cv2.inRange(hsv_image, orange_lower, orange_upper), label_region_mask)
        cherry_mask = cv2.bitwise_and(cv2.inRange(hsv_image, cherry_lower, cherry_upper), label_region_mask)

        orange_pixels = cv2.countNonZero(orange_mask)
        cherry_pixels = cv2.countNonZero(cherry_mask)

        fruit_type = "香蕉"
        if orange_pixels > 140 and cherry_pixels < 300:
            fruit_type = "橙子"
        elif orange_pixels > 140 and cherry_pixels > 300:
            fruit_type = "樱桃"

        result["水果类型"] = fruit_type

        def get_centroid(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] == 0:
                return None
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)

        fruit_center = None
        if fruit_type == "橙子":
            fruit_center = get_centroid(orange_mask)
        elif fruit_type == "樱桃":
            fruit_center = get_centroid(cherry_mask)
        elif fruit_type == "香蕉":
            fruit_mask = (sam_image == 124).astype(np.uint8)
            fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
            fruit_center = get_centroid(fruit_mask)

        if fruit_center:
            fx, fy = fruit_center
            lx, ly = label_top_left
            dx = abs(fx - lx)
            dy = abs(fy - ly)
            result["水果偏移_x"] = dx
            result["水果偏移_y"] = dy

            if fruit_type == "橙子":
                result["橙子_x偏移"] = dx
                result["橙子_y偏移"] = dy
            elif fruit_type == "樱桃":
                result["樱桃_x偏移"] = dx
                result["樱桃_y偏移"] = dy
            elif fruit_type == "香蕉":
                result["香蕉_x偏移"] = dx
                result["香蕉_y偏移"] = dy

            if fruit_type in fruit_offset_ranges:
                x_range = fruit_offset_ranges[fruit_type]["x"]
                y_range = fruit_offset_ranges[fruit_type]["y"]

                if not (x_range[0] <= dx <= x_range[1]) or not (y_range[0] <= dy <= y_range[1]):
                    result["水果偏移OK"] = False
                    reasons.append(f"{fruit_type} 水果偏移 ({dx}, {dy}) 不在允许范围内")
        else:
            result["水果偏移OK"] = False
            reasons.append(f"{fruit_type} 水果中心点未找到")
            result["水果偏移_x"] = np.nan
            result["水果偏移_y"] = np.nan

        if fruit_type in hsv_ranges:
            hsv_range = hsv_ranges[fruit_type]
            _, fruit_length = get_largest_region_and_height(hsv_image, hsv_range)

            if fruit_length is None:
                result["水果长度OK"] = False
                reasons.append(f"{fruit_type} 水果区域未找到")
            else:
                result["水果区域长度"] = fruit_length
                if fruit_length > 310 or fruit_length < 283:
                    result["水果长度OK"] = False
                    reasons.append(f"{fruit_type} 区域长度 {fruit_length} > 310")

    if len(valid_label_contours) == 2:
        rect1, rect2 = [cv2.boundingRect(cnt) for cnt in valid_label_contours]
        left_diff = abs(rect1[0] - rect2[0])
        right_diff = abs((rect1[0] + rect1[2]) - (rect2[0] + rect2[2]))
        if left_diff > 37 or right_diff > 37:
            result["标签纸对齐OK"] = False
            reasons.append(f"标签纸左右上角未对齐，左差 {left_diff}，右差 {right_diff}")

        widths = [cv2.boundingRect(cnt)[2] for cnt in valid_label_contours]
        if any(w < c for w in widths):
            result["标签纸宽度OK"] = False
            reasons.append(f"存在标签纸宽度小于 {c}")

        for idx, cnt in enumerate(valid_label_contours):
            area = cv2.contourArea(cnt)
            rotated_rect = cv2.minAreaRect(cnt)
            rect_width, rect_height = rotated_rect[1]
            rect_area = rect_width * rect_height
            rectangularity = area / rect_area if rect_area != 0 else 0
            # rectangularity = rect_area - area 
            if rectangularity < 0.914:
                result["标签纸矩形度OK"] = False
                reasons.append(f"标签纸矩形度 {rectangularity:.2f} > 0.92")


        if len(valid_label_contours) == 2:
            area1 = cv2.contourArea(valid_label_contours[0])
            area2 = cv2.contourArea(valid_label_contours[1])
            small_label_contour = valid_label_contours[0] if area1 < area2 else valid_label_contours[1]

            x, y, w, h = cv2.boundingRect(small_label_contour)

            left_right_shrink = 27
            top_bottom_shrink = 11

            new_x = x + left_right_shrink
            new_y = y + top_bottom_shrink
            new_w = w - 2 * left_right_shrink
            new_h = h - 2 * top_bottom_shrink

            if new_w > 0 and new_h > 0:
                small_label_mask = np.zeros_like(sam_image)
                cv2.rectangle(small_label_mask, (new_x, new_y), (new_x + new_w, new_y + new_h), 255, thickness=cv2.FILLED)
            else:
                small_label_mask = np.zeros_like(sam_image)

            v_channel = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)[:, :, 2]
            v_threshold_mask = cv2.inRange(v_channel, 0,190)
            combined_mask = cv2.bitwise_and(v_threshold_mask, small_label_mask)

            contours_v, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_v) > 1:
                total_area = sum(cv2.contourArea(cnt) for cnt in contours_v)
                result["小标签V通道总面积"] = total_area
                if total_area < 1800:
                    result["小标签V通道面积和NG"] = True
                    reasons.append(f"小标签V通道剩余区域总面积 {total_area} < 1800")
            else:
                result["小标签V通道总面积"] = 0
                result["小标签V通道面积和NG"] = True
                reasons.append("小标签V通道无足够区域")

    fruit_mask = (sam_image == 124).astype(np.uint8)
    fruit_contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_fruit_contours = [cnt for cnt in fruit_contours if cv2.contourArea(cnt) > min_area]

    for cnt in valid_fruit_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.9:
            result["水果圆度OK"] = False
            reasons.append(f"水果区域圆度 {circularity:.2f} > 0.9")

    fruit_type = result["水果类型"]
    liquid_type = result["液体类型"]

    mapping = {
        "樱桃": "樱桃水",
        "橙子": "橙子水",
        "香蕉": "香蕉水"
    }

    if fruit_type in mapping and mapping[fruit_type] != liquid_type:
        result["水果液体匹配OK"] = False
        reasons.append(f"水果类型 '{fruit_type}' 与液体类型 '{liquid_type}' 不匹配")

    if not all([
        result["液体区域OK"],
        result["液体长度OK"],
        result["标签纸数量OK"],
        result["标签纸对齐OK"],
        result["标签纸宽度OK"],
        result["水果圆度OK"],
        result["标签纸矩形度OK"],
        result["水果液体匹配OK"],
        result["水果偏移OK"],
        result["水果长度OK"],
        result["标签纸面积顺序OK"],
        result["标签纸间距OK"],
        not result["小标签V通道面积和NG"],

    ]):
        result["Overall"] = "NG"
    else:
        reasons.append("无异常")

    if result["Overall"] == "NG":
        anomaly_flag = True
    else:
        anomaly_flag = False

    return anomaly_flag
