import os
import sys
from .mz_stylize_photo_utils import Utils
from nodes import MAX_RESOLUTION
import comfy.utils
import shutil
import folder_paths
import comfy.samplers


AUTHOR_NAME = u"[MinusZone]"
CATEGORY_NAME = f"{AUTHOR_NAME} StylizePhoto"

sys.path.append(os.path.join(os.path.dirname(__file__)))

import importlib


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}



import importlib
import mz_stylize_photo_core
class MZ_StylizePhotoKSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        STYLE_TYPE = [
            "clay.v1"
        ] 
        return {
            "required":{
                "xl_ckpt_name": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": -1, "min": -1, "max": MAX_RESOLUTION}),
                "style_type": (STYLE_TYPE,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
            },
            "optional":{
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "control_net": ("StylizePhoto_CONTROL_NET",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ksampler"

    CATEGORY = CATEGORY_NAME

    
    def ksampler(self, **kwargs):
        importlib.reload(mz_stylize_photo_core)
        return mz_stylize_photo_core.ksampler(kwargs)

                 

NODE_CLASS_MAPPINGS["MZ_StylizePhotoKSamplerNode"] = MZ_StylizePhotoKSamplerNode
NODE_DISPLAY_NAME_MAPPINGS["MZ_StylizePhotoKSamplerNode"] = f"{AUTHOR_NAME} - StylizePhotoKSampler"


class MZ_StylizePhotoControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        importlib.reload(mz_stylize_photo_core)
        required = {}
        for key, _ in mz_stylize_photo_core.CONTROLNET_TYPES.items():
            required[
                f"{key}_control_net_name"
            ] = (["none"] + folder_paths.get_filename_list("controlnet"),)
 

        return {"required": required}

    RETURN_TYPES = ("StylizePhoto_CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = CATEGORY_NAME

    def load_controlnet(self, **kwargs):
        importlib.reload(mz_stylize_photo_core)
        return mz_stylize_photo_core.load_controlnet(kwargs)
    


NODE_CLASS_MAPPINGS["MZ_StylizePhotoControlNetApply"] = MZ_StylizePhotoControlNetApply
NODE_DISPLAY_NAME_MAPPINGS["MZ_StylizePhotoControlNetApply"] = f"{AUTHOR_NAME} - StylizePhotoControlNetApply"