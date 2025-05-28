import os
from typing import Union, List, OrderedDict
import torch
import numpy as np
import torch.nn as nn


obj_list = [
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
    "object",
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni",
    "pcb",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

mvtec_obj_list = [
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
]

visa_obj_list = [
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
]

mvtec_anomaly_detail = {
    "carpet": "different color,cut,hole,metal contamination,thread",
    "grid": "bent,broken,glue,metal contamination,thread",
    "leather": "different color,cut,fold,glue,poke",
    "tile": "crack,gule strip,gray stroke,oil,rough",
    "wood": "different color,combined,hole,liquid,scratch",
    "bottle": "large broken,small broken,contamination",
    "cable": "bent wire,cable swap,combined,cut inner insulation,cut outer insulation,missing cable,missing wire,poke insulation",
    "capsule": "crack,faulty imprint,poke,scratch,squeeze",
    "hazelnut": "crack,cut,hole,print",
    "metal nut": "bent,different color,flip,scratch",
    "pill": "different color,combined,contamination,crack,faulty imprint,pill type,scratch",
    "screw": "manipulated front,scratch head,scratch neck,thread side,thread top",
    "toothbrush": "defective",
    "transistor": "bent lead,cut lead,damaged case,misplaced",
    "zipper": "broken teeth,combined,fabric border,fabric interior,rough,split teeth,squeezed teeth",
}

visa_anomaly_detail = {
    "candle": "chunk of wax missing,foreign particals on candle,different colour spot,damaged corner of packaging,weird candle wick,other,extra wax in candle,wax melded out of the candle",
    "capsules": "bubble,discolor,scratch,leak,misshape",
    "cashew": "burnt,corner or edge breakage,different colour spot,middle breakage,small holes,same colour spot,small scratches,stuck together",
    "chewinggum": "chunk of gum missing,scratches,small cracks,corner missing,similar colour spot",
    "fryum": "burnt,similar colour spot,corner or edge breakage,middle breakage,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "chip around edge and corner,small scratches,small cracks,different colour spot,middle breakage,similar colour spot",
    "macaroni2": "breakage down the middle,small scratches,color spot similar to the object,different color spot,small chip around edge,small cracks,other",
    "pcb1": "bent,melt,scratch,missing",
    "pcb2": "bent,melt,scratch,missing",
    "pcb3": "bent,melt,scratch,missing",
    "pcb4": "burnt,scratch,dirt,damage,extra,missing,wrong place",
    "pipe fryum": "burnt,corner and edge breakage,different colour spot,middle breakage,small scratches,small cracks,similar colour spot,stuck together",
}

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_abnormal_detail = {}
for cls_name in mvtec_obj_list:
    prompt_abnormal_detail[cls_name] = prompt_abnormal + ['normal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail[cls_name].split(',')]

for cls_name in visa_obj_list:
    prompt_abnormal_detail[cls_name] = prompt_abnormal + ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail[cls_name].split(',')]


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts
