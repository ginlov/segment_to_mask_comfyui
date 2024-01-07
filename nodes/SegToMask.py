import torch
import numpy as np

from comfy.model_management import InterruptProcessingException, get_torch_device
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

LABEL2ID = {
    "airplane": 90,
    "animal": 126,
    "apparel": 92,
    "arcade machine": 78,
    "armchair": 30,
    "ashcan": 138,
    "awning": 86,
    "bag": 115,
    "ball": 119,
    "bannister": 95,
    "bar": 77,
    "barrel": 111,
    "base": 40,
    "basket": 112,
    "bathtub": 37,
    "bed ": 7,
    "bench": 69,
    "bicycle": 127,
    "blanket": 131,
    "blind": 63,
    "boat": 76,
    "book": 67,
    "bookcase": 62,
    "booth": 88,
    "bottle": 98,
    "box": 41,
    "bridge": 61,
    "buffet": 99,
    "building": 1,
    "bulletin board": 144,
    "bus": 80,
    "cabinet": 10,
    "canopy": 106,
    "car": 20,
    "case": 55,
    "ceiling": 5,
    "chair": 19,
    "chandelier": 85,
    "chest of drawers": 44,
    "clock": 148,
    "coffee table": 64,
    "column": 42,
    "computer": 74,
    "conveyer belt": 105,
    "counter": 45,
    "countertop": 70,
    "cradle": 117,
    "crt screen": 141,
    "curtain": 18,
    "cushion": 39,
    "desk": 33,
    "dirt track": 91,
    "dishwasher": 129,
    "door": 14,
    "earth": 13,
    "escalator": 96,
    "fan": 139,
    "fence": 32,
    "field": 29,
    "fireplace": 49,
    "flag": 149,
    "floor": 3,
    "flower": 66,
    "food": 120,
    "fountain": 104,
    "glass": 147,
    "grandstand": 51,
    "grass": 9,
    "hill": 68,
    "hood": 133,
    "house": 25,
    "hovel": 79,
    "kitchen island": 73,
    "lake": 128,
    "lamp": 36,
    "land": 94,
    "light": 82,
    "microwave": 124,
    "minibike": 116,
    "mirror": 27,
    "monitor": 143,
    "mountain": 16,
    "ottoman": 97,
    "oven": 118,
    "painting": 22,
    "palm": 72,
    "path": 52,
    "person": 12,
    "pier": 140,
    "pillow": 57,
    "plant": 17,
    "plate": 142,
    "plaything": 108,
    "pole": 93,
    "pool table": 56,
    "poster": 100,
    "pot": 125,
    "radiator": 146,
    "railing": 38,
    "refrigerator": 50,
    "river": 60,
    "road": 6,
    "rock": 34,
    "rug": 28,
    "runway": 54,
    "sand": 46,
    "sconce": 134,
    "screen": 130,
    "screen door": 58,
    "sculpture": 132,
    "sea": 26,
    "seat": 31,
    "shelf": 24,
    "ship": 103,
    "shower": 145,
    "sidewalk": 11,
    "signboard": 43,
    "sink": 47,
    "sky": 2,
    "skyscraper": 48,
    "sofa": 23,
    "stage": 101,
    "stairs": 53,
    "stairway": 59,
    "step": 121,
    "stool": 110,
    "stove": 71,
    "streetlight": 87,
    "swimming pool": 109,
    "swivel chair": 75,
    "table": 15,
    "tank": 122,
    "television receiver": 89,
    "tent": 114,
    "toilet": 65,
    "towel": 81,
    "tower": 84,
    "trade name": 123,
    "traffic light": 136,
    "tray": 137,
    "tree": 4,
    "truck": 83,
    "van": 102,
    "vase": 135,
    "wall": 0,
    "wardrobe": 35,
    "washer": 107,
    "water": 21,
    "waterfall": 113,
    "windowpane": 8
  }

class PipelineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    CATEGORY = "giangvlcs/PipelineLoader"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "PROCESSOR")
    RETURN_TYPES = ("MODEL", "MODEL")

    def load(self):
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        return image_segmentor, image_processor

class SegToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "classes": ("STRING", {"multiline": True}), 
                "image": ("IMAGE",),
                "model": ("MODEL",), 
                "processor": ("MODEL",)
            },
        }
        
    CATEGORY = "giangvlcs/segtomask"
    TITLE = "Segmentation to Mask"
    RETURN_TYPES = ("IMAGE", "MASK")
    RUTURN_NAMES = ("image", "mask")
    FUNCTION = "segment2mask"

    def segment2mask(self, classes, image, model, processor):
        classes = classes.split(",")
        classes = [each.strip() for each in classes]
        temp = image[0].cpu().numpy() * 255.0
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        img = Image.fromarray(temp).convert("RGB")
        # img = Image.fromarray(image.detach().cpu().numpy()).convert("RGB")

        pixel_values = processor(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = model(pixel_values)

        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
        color_seg = np.ones((seg.shape[0], seg.shape[1], 4), dtype=np.uint8) * 255
        list_of_ids = [LABEL2ID[label] for label in classes]
        masks = []
        temp1 = np.array(img.convert('RGBA'))
        temp2 = temp1
        temp2[:, :, -1] = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
        total_mask = temp1[:, :, -1]
        for each_id in list_of_ids:
            mask = seg == each_id
            masks.append(mask)
            # color_seg[mask, :] = temp1[mask, :]

        original_mask = True
        for mask in masks:
            original_mask = original_mask & ~mask
        # color_seg[original_mask, :] = temp2[original_mask, :]
        total_mask[original_mask] = 0
        total_mask = torch.tensor(total_mask, dtype=torch.float32) / 255.0
        total_mask = 1. - total_mask

        # color_seg = torch.from_numpy(color_seg.astype(np.float32) / 255.0)[None, ]
        return (image, [total_mask])
