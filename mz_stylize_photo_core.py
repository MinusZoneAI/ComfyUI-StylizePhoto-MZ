
import os

import cv2
from nodes import common_ksampler,VAEEncode,VAEDecode,CheckpointLoaderSimple,ControlNetLoader,ControlNetApplyAdvanced
import importlib 


import mz_stylize_photo_utils

# importlib.reload(mz_stylize_photo_utils)



FIXED_QUALITY_PROMPT = "(high quality), (best quality), (masterpiece), (8K resolution), (2k wallpaper)"

STYLE_PROMPTS = {
    "clay.v1": {
        # "positive": "clay_world,clay_style,plasticine_style,photography,macro,tilt shift,cute,by Adult Swim,",
        "positive": "clay,clay_model,photography,macro,cute,by Makoto Shinkai,",
        "negative": "blurry,noisy,text,watermark"
    }
}

STYLE_TYPE_LORA_INFOS = {
    "clay.v1": {
        "url": "https://www.modelscope.cn/api/v1/models/wailovet/MinusZoneAIModels/repo?Revision=master&FilePath=stylize_photo_models%2Fclay_v1.pt",
        "output": "stylize_photo_models/clay_v1.pt",
    }
}

Utils = mz_stylize_photo_utils.Utils

def ksampler(kwargs):
    image = kwargs.get("image")
    resolution = kwargs.get("resolution")
    style_type = kwargs.get("style_type")

    ckpt_name = kwargs.get("xl_ckpt_name", None)
    if ckpt_name is not None and ckpt_name != "none":
        cache_key = f"checkpoints_{ckpt_name}"
        mcv = Utils.cache_get(cache_key)
        if mcv is not None:
            print("Using cached model, clip, vae")
            model, clip, vae = mcv
        else:
            model, clip, vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
            Utils.cache_set(cache_key, (model, clip, vae))
    else:
        model = kwargs.get("model")
        clip = kwargs.get("clip")
        vae = kwargs.get("vae")

    seed = kwargs.get("seed")
    steps = kwargs.get("steps")
    cfg = kwargs.get("cfg")
    denoise = kwargs.get("denoise")
    positive_prompt = kwargs.get("positive_prompt")
    negative_prompt = kwargs.get("negative_prompt")

    sampler_name = "euler_ancestral"
    scheduler = "normal"

 
    if resolution > 0:
        image_pil = Utils.tensor2pil(image)
        image_pil = Utils.resize_max(image_pil, resolution, resolution)
        image = Utils.pil2tensor(image_pil)
        image = Utils.list_tensor2tensor([image])

    latent_image = VAEEncode().encode(vae, image)[0]
    
    lora_info = STYLE_TYPE_LORA_INFOS[style_type]
    lora_path = Utils.download_model(lora_info)


    
    model = Utils.load_lora(model, lora_path, 0.88)



    positive = Utils.native_clip_text_encode(clip, f"{FIXED_QUALITY_PROMPT},{STYLE_PROMPTS[style_type]['positive']},{positive_prompt}")
    negative = Utils.native_clip_text_encode(clip, f"{STYLE_PROMPTS[style_type]['negative']},{negative_prompt}")


    control_net = kwargs.get("control_net", {})
    for key, item in control_net.items():
        print(f"Applying controlnet {key}")
        preprocessed_func = CONTROLNET_TYPES[key].get("preprocessing", None)
        preprocessed_image = Utils.tensor2pil(image)
        if preprocessed_func is not None:
            preprocessed_image = preprocessed_func(preprocessed_image)
            preprocessed_image = Utils.pil2tensor(preprocessed_image)
            preprocessed_image = Utils.list_tensor2tensor([preprocessed_image])
        
        strength = CONTROLNET_TYPES[key].get("strength", 1.0)
        start_percent = CONTROLNET_TYPES[key].get("start_percent", 0.0)
        end_percent = CONTROLNET_TYPES[key].get("end_percent", 1)
        positive, negative = ControlNetApplyAdvanced().apply_controlnet(positive, negative, item, preprocessed_image, strength, start_percent, end_percent)
        

    latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]

    image = VAEDecode().decode(vae, latent)[0]
    return (image,)



from PIL import ImageFilter


def preprocessing_canny(image): 
    # image = Utils.resize_max(image, 512, 512)
    # image = image.filter(ImageFilter.FIND_EDGES)

    image = Utils.pil2cv(image)
    image = cv2.Canny(image, 100, 200)
    image = Utils.cv2pil(image)

    # image.save("canny.jpg")
    return image


def preprocessing_tile(image):
    return Utils.resize_max(image, 512, 512)

CONTROLNET_TYPES = {
    "tile": {
        "preprocessing": preprocessing_tile,
        "strength": 0.35,
        "start_percent": 0.0,
        "end_percent": 1,
    },
    "canny": {
        "preprocessing": preprocessing_canny,
        "strength": 0.45,
        "start_percent": 0.0,
        "end_percent": 1,
    },
}

def load_controlnet(kwargs):
    result = {}
    for key, _ in CONTROLNET_TYPES.items():
        control_net_name = kwargs.get(f"{key}_control_net_name", None)
        if control_net_name is not None and control_net_name != "none":
            result[key] = ControlNetLoader().load_controlnet(control_net_name)[0]
             
    return (result,)