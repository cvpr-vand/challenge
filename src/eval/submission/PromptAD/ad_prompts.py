

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
    "breakfast_box": "breakfast box",
    "juice_bottle": "juice bottle",
    "screw_bag": "screw bag",
    "splicing_connectors": "splicing connectors",
}


state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

#
class_state_abnormal = {
    'bottle': ['{} with large breakage', '{} with small breakage', '{} with contamination'],
    'toothbrush': ['{} with defect', '{} with anomaly'],
    'carpet': ['{} with hole', '{} with color stain', '{} with metal contamination', '{} with thread residue', '{} with thread', '{} with cut'],
    'hazelnut': ['{} with crack', '{} with cut', '{} with hole', '{} with print'],
    'leather': ['{} with color stain', '{} with cut', '{} with fold', '{} with glue', '{} with poke'],
    'cable': ['{} with bent wire', '{} with missing part', '{} with missing wire', '{} with cut', '{} with poke'],
    'capsule': ['{} with crack', '{} with faulty imprint', '{} with poke', '{} with scratch', '{} squeezed with compression'],
    'grid': ['{} with breakage',  '{} with thread residue', '{} with thread', '{} with metal contamination', '{} with glue', '{} with a bent shape'],
    'pill': ['{} with color stain', '{} with contamination', '{} with crack', '{} with faulty imprint', '{} with scratch', '{} with abnormal type'],
    'transistor': ['{} with bent lead', '{} with cut lead', '{} with damage', '{} with misplaced transistor'],
    'metal_nut': ['{} with a bent shape ', '{} with color stain', '{} with a flipped orientation', '{} with scratch'],
    'screw': ['{} with manipulated front',  '{} with scratch neck', '{} with scratch head'],
    'zipper': ['{} with broken teeth', '{} with fabric border', '{} with defect fabric', '{} with broken fabric', '{} with split teeth', '{} with squeezed teeth'],
    'tile': ['{} with crack', '{} with glue strip', '{} with gray stroke', '{} with oil', '{} with rough surface'],
    'wood': ['{} with color stain', '{} with hole', '{} with scratch', '{} with liquid'],

    'candle': ['{} with melded wax', '{} with foreign particals', '{} with extra wax', '{} with chunk of wax missing', '{} with weird candle wick', '{} with damaged corner of packaging', '{} with different colour spot'],
    'capsules': ['{} with scratch', '{} with discolor', '{} with misshape', '{} with leak', '{} with bubble'],
    # 'capsules': [],
    'cashew': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with spot'],
    'chewinggum': ['{} with corner missing', '{} with scratches', '{} with chunk of gum missing', '{} with colour spot', '{} with cracks'],
    'fryum': ['{} with breakage', '{} with scratches', '{} with burnt', '{} with colour spot', '{} with fryum stuck together', '{} with colour spot'],
    'macaroni1': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'macaroni2': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'pcb1': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb2': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb3': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb4': ['{} with scratch', '{} with extra', '{} with missing', '{} with wrong place', '{} with damage', '{} with burnt', '{} with dirt'],
    'pipe_fryum': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with colour spot', '{} with cracks'],
    
    'breakfast_box': ['{} with structural anomalies', '{} with logical anomalies'],
    'juice_bottle': ['{} with structural anomalies', '{} with logical anomalies'],
    'pushpins': ['{} with structural anomalies', '{} with logical anomalies'],
    'screw_bag': ['{} with structural anomalies', '{} with logical anomalies'],
    # 'splicing_connectors': ['{} with structural anomalies', '{} with logical anomalies']
    # 'breakfast_box': ['{} with missing almonds', '{} with missing bananas', '{} with missing toppings', '{} with missing cereals', 
    #                 '{} with missing cereals and toppings', '{} with 2 nectarines 1 tangerine', '{} with 1 nectarine 1 tangerine', 
    #                 '{} with 0 nectarines 2 tangerines', '{} with 0 nectarines 3 tangerines', '{} with 3 nectarines 0 tangerines', 
    #                 '{} with 0 nectarines 1 tangerine', '{} with 0 nectarines 0 tangerines', '{} with 0 nectarines 4 tangerines', 
    #                 '{} with compartments swapped', '{} with overflow', '{} with underflow', '{} with wrong ratio', '{} with mixed cereals', 
    #                 '{} with fruit damaged', '{} with box damaged', '{} with toppings crushed', '{} with contamination'],
    # 'juice_bottle': ['{} with missing top label', '{} with missing bottom label', '{} with swapped labels', '{} with damaged label', 
    #                 '{} with rotated label', '{} with misplaced label top', '{} with misplaced label bottom', '{} with label text incomplete', 
    #                 '{} with empty bottle', '{} with wrong fill level too much', '{} with wrong fill level not enough', '{} with misplaced fruit icon', 
    #                 '{} with missing fruit icon', '{} with unknown fruit icon', '{} with incomplete fruit icon', '{} with wrong juice type', '{} with juice color', 
    #                 '{} with contamination'],
    # 'pushpins': ['{} with 1 additional pushpin', '{} with 2 additional pushpins', '{} with missing pushpin', '{} with missing separator', '{} with front bent', 
    #             '{} with broken', '{} with color', '{} with contamination'],
    # 'screw_bag': ['{} with screw too long', '{} with screw too short', '{} with 1 very short screw', '{} with 2 very short screws', '{} with 1 additional long screw', 
    #             '{} with 1 additional short screw', '{} with 1 additional nut', '{} with 2 additional nuts', '{} with 1 additional washer', '{} with 2 additional washers', 
    #             '{} with 1 missing long screw', '{} with 1 missing short screw', '{} with 1 missing nut', '{} with 2 missing nuts', '{} with 1 missing washer', 
    #             '{} with 2 missing washers', '{} with bag broken', '{} with color', '{} with contamination', '{} with part broken'],
    'splicing_connectors': ['{} with wrong connector type 5 to 2', '{} with wrong connector type 5 to 3', '{} with wrong connector type 3 to 2', '{} with cable too short T2', 
                        '{} with cable too short T3', '{} with cable too short T5', '{} with missing connector', '{} with missing connector and cable', '{} with missing cable', 
                        '{} with extra cable', '{} with cable color', '{} with broken cable', '{} with cable cut', '{} with cable not plugged', '{} with unknown cable color', 
                        '{} with wrong cable location', '{} with flipped connector', '{} with broken connector', '{} with open lever', '{} with color', '{} with contamination']
    }
