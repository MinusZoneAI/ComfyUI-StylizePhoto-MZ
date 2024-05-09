
import os
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
import folder_paths
import base64
from PIL import Image, ImageFilter
import io
import torch
import re
import hashlib
import cv2
# sys.path.append(os.path.join(os.path.dirname(__file__)))
temp_directory = folder_paths.get_temp_directory()
from tqdm import tqdm
import requests
import comfy.utils


CACHE_POOL = {}


class Utils:
    def Md5(str):
        return hashlib.md5(str.encode('utf-8')).hexdigest()

    def check_frames_path(frames_path):

        if frames_path == "" or frames_path.startswith(".") or frames_path.startswith("/") or frames_path.endswith("/") or frames_path.endswith("\\"):
            return "frames_path不能为空"

        frames_path = os.path.join(
            folder_paths.get_output_directory(), frames_path)

        if frames_path == folder_paths.get_output_directory():
            return "frames_path不能为output目录"

        return ""

    def base64_to_pil_image(base64_str):
        if base64_str is None:
            return None
        if len(base64_str) == 0:
            return None
        if type(base64_str) not in [str, bytes]:
            return None
        if base64_str.startswith("data:image/png;base64,"):
            base64_str = base64_str.split(",")[-1]
        base64_str = base64_str.encode("utf-8")
        base64_str = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(base64_str))

    def pil_image_to_base64(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = str(img_str, encoding="utf-8")
        return f"data:image/png;base64,{img_str}"

    def listdir_png(path):
        try:
            files = os.listdir(path)
            new_files = []
            for file in files:
                if file.endswith(".png"):
                    new_files.append(file)
            files = new_files
            files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            return files
        except Exception as e:
            return []

    def tensor2pil(image):
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Convert PIL to Tensor

    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)[0]

    def pil2cv(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def cv2pil(image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def list_tensor2tensor(data):
        result_tensor = torch.stack(data)
        return result_tensor

    def loadImage(path):
        img = Image.open(path)
        img = img.convert("RGB")
        return img

    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def native_vae_encode(vae, image):
        pixels = Utils.vae_encode_crop_pixels(image)
        t = vae.encode(pixels[:, :, :, :3])
        return {"samples": t}

    def native_vae_encode_for_inpaint(vae, pixels, mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape(
            (-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        # grow mask by a few pixels to keep things seamless in latent space

        mask_erosion = mask

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        t = vae.encode(pixels)

        return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}

    def native_vae_decode(vae, samples):
        return vae.decode(samples["samples"])

    def native_clip_text_encode(clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def cache_get(key):
        return CACHE_POOL.get(key, None)

    def cache_set(key, value):
        global CACHE_POOL
        CACHE_POOL[key] = value
        return True

    def get_models_path():
        models_path = os.path.join(
            folder_paths.models_dir, "minus_zone_models")
        os.makedirs(models_path, exist_ok=True)
        return models_path

    def translate_text(text, from_code, to_code):
        try:
            import argostranslate
            from argostranslate import translate
        except ImportError:
            subprocess.run([
                sys.executable, "-m",
                "pip", "install", "argostranslate"], check=True)
            import argostranslate
            from argostranslate import translate

        try:
            translation = translate.get_translation_from_codes(
                from_code, to_code)
            if translation is None:
                raise Exception("Translation not found")

        except Exception as e:
            print(e)
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(
                    lambda x: (x.from_code == from_code and x.to_code ==
                               to_code), available_packages,
                )
            )
            download_path = package_to_install.download()
            print("package_to_install.download():", download_path)
            argostranslate.package.install_from_path(download_path)

            translation = translate.get_translation_from_codes(
                from_code, to_code)
            if translation is None:
                return text

        # Translate
        translatedText = translation.translate(
            text)

        return translatedText

    def zh2en(text):
        return Utils.translate_text(text, "zh", "en")

    def en2zh(text):
        return Utils.translate_text(text, "en", "zh")

    def prompt_zh_to_en(prompt):
        prompt = prompt.replace("，", ",")
        prompt = prompt.replace("。", ",")
        prompt = prompt.replace("\n", ",")
        tags = prompt.split(",")
        # 判断是否有中文
        for i, tag in enumerate(tags):
            if re.search(u'[\u4e00-\u9fff]', tag):
                tags[i] = Utils.zh2en(tag)
                # 如果第一个字母是大写,转为小写
                if tags[i][0].isupper():
                    tags[i] = tags[i].lower().replace(".", "")

        return ",".join(tags)

    def mask_resize(mask, width, height):
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask, size=(height, width), mode="bilinear")
        mask = mask.squeeze(0).squeeze(0)
        return mask

    def mask_threshold(interested_mask):
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        ret, thresh1 = cv2.threshold(
            mask_image_cv2, 127, 255, cv2.THRESH_BINARY)
        thresh1 = Utils.cv2pil(thresh1)
        thresh1 = np.array(thresh1)
        thresh1 = thresh1[:, :, 0]
        return Utils.pil2tensor(thresh1)

    def mask_erode(interested_mask, value):
        value = int(value)
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask_image_cv2, kernel, iterations=value)
        erosion = Utils.cv2pil(erosion)
        erosion = np.array(erosion)
        erosion = erosion[:, :, 0]
        return Utils.pil2tensor(erosion)

    def mask_dilate(interested_mask, value):
        value = int(value)
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(mask_image_cv2, kernel, iterations=value)
        dilation = Utils.cv2pil(dilation)
        dilation = np.array(dilation)
        dilation = dilation[:, :, 0]
        return Utils.pil2tensor(dilation)

    def mask_edge_opt(interested_mask, edge_feathering):

        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)

        # 高斯模糊
        dilation2 = Utils.cv2pil(mask_image_cv2)
        dilation2 = mask_image.filter(
            ImageFilter.GaussianBlur(edge_feathering))

        # mask_image dilation2 图片蒙版叠加
        dilation2 = Utils.pil2cv(dilation2)
        # dilation2[mask_image_cv2 < 127] = 0
        dilation2 = Utils.cv2pil(dilation2)
        # to RGB
        dilation2 = np.array(dilation2)
        dilation2 = dilation2[:, :, 0]
        return Utils.pil2tensor(dilation2)

    def mask_composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(
                destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

        x = max(-source.shape[3] * multiplier,
                min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier,
                min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape(
                (-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

        # calculate the bounds of the source that will be overlapping the destination
        # this prevents the source trying to overwrite latent pixels that are out of bounds
        # of the destination
        visible_width, visible_height = (
            destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask * \
            destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom,
                    left:right] = source_portion + destination_portion
        return destination

    def latent_upscale_by(samples, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], width, height, "nearest-exact", "disabled")
        return s

    def resize_by(image, percent):
        # 判断类型是否为PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        width, height = image.size
        new_width = int(width * percent)
        new_height = int(height * percent)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def resize_max(im, dst_w, dst_h):
        src_w, src_h = im.size

        if src_h < src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        return im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

    def get_device():
        return comfy.model_management.get_torch_device()

    def download_file(url, filepath, threads=8, retries=6):

        get_size_tmp = requests.get(url, stream=True)
        total_size = int(get_size_tmp.headers.get("content-length", 0))

        print(f"Downloading {url} to {filepath} with size {total_size} bytes")

        base_filename = os.path.basename(filepath)
        cache_dir = os.path.join(os.path.dirname(
            filepath), f"{base_filename}.t_{threads}_cache")
        os.makedirs(cache_dir, exist_ok=True)

        def get_total_existing_size():
            fs = os.listdir(cache_dir)
            existing_size = 0
            for f in fs:
                if f.startswith("block_"):
                    existing_size += os.path.getsize(
                        os.path.join(cache_dir, f))
            return existing_size

        total_existing_size = get_total_existing_size()

        if total_size != 0 and total_existing_size != total_size:

            with tqdm(total=total_size, initial=total_existing_size, unit="B", unit_scale=True) as progress_bar:
                all_threads = []

                for i in range(threads):
                    cache_filepath = os.path.join(cache_dir, f"block_{i}")

                    start = total_size // threads * i
                    end = total_size // threads * (i + 1) - 1

                    if i == threads - 1:
                        end = total_size

                    # Check if the file already exists
                    if os.path.exists(cache_filepath):
                        # Get the size of the existing file
                        existing_size = os.path.getsize(cache_filepath)
                    else:
                        existing_size = 0

                    headers = {"Range": f"bytes={start + existing_size}-{end}"}
                    if end == total_size:
                        headers = {"Range": f"bytes={start + existing_size}-"}
                    if start + existing_size >= end:
                        continue
                    # print(f"Downloading {cache_filepath} with headers bytes={start + existing_size}-{end}")

                    # Streaming, so we can iterate over the response.
                    response = requests.get(url, stream=True, headers=headers)

                    def download_file_thread(response, cache_filepath):
                        block_size = 1024
                        if end - (start + existing_size) < block_size:
                            block_size = end - (start + existing_size)
                        with open(cache_filepath, "ab") as file:
                            for data in response.iter_content(block_size):
                                file.write(data)
                                progress_bar.update(
                                    len(data)
                                )

                    t = threading.Thread(
                        target=download_file_thread, args=(response, cache_filepath))

                    all_threads.append(t)

                    t.start()

                for t in all_threads:
                    t.join()

            if total_size != 0 and get_total_existing_size() > total_size:
                # 文件下载失败
                shutil.rmtree(cache_dir)
                raise RuntimeError("Download failed, file is incomplete")

            if total_size != 0 and total_size != get_total_existing_size():
                if retries > 0:
                    retries -= 1
                    print(
                        f"Download failed: {total_size} != {get_total_existing_size()}, retrying... {retries} retries left")
                    return Utils.download_file(url, filepath, threads, retries)

                # 文件损坏
                raise RuntimeError(
                    f"Download failed: {total_size} != {get_total_existing_size()}")

        if os.path.exists(filepath):
            shutil.move(filepath, filepath + ".old." +
                        time.strftime("%Y%m%d%H%M%S"))

        # merge the files
        with open(filepath, "wb") as f:
            for i in range(threads):
                cache_filepath = os.path.join(cache_dir, f"block_{i}")
                with open(cache_filepath, "rb") as cf:
                    f.write(cf.read())

        shutil.rmtree(cache_dir)
        return filepath

    def hf_download_model(url, only_get_path=False):
        if not url.startswith("https://"):
            raise ValueError("URL must start with https://")
        if url.startswith("https://huggingface.co/") or url.startswith("https://hf-mirror.com/"):
            base_model_path = os.path.abspath(os.path.join(
                Utils.get_models_path(), "transformers_models"))
            # https://huggingface.co/FaradayDotDev/llama-3-8b-Instruct-GGUF/resolve/main/llama-3-8b-Instruct.Q2_K.gguf?download=true
            texts = url.split("?")[0].split("/")
            file_name = texts[-1]
            zone_path = f"{texts[3]}/{texts[4]}"

            save_path = os.path.join(base_model_path, zone_path, file_name)

            if os.path.exists(save_path) is False:
                if only_get_path:
                    return None
                os.makedirs(os.path.join(
                    base_model_path, zone_path), exist_ok=True)
                Utils.download_file(url, save_path)

            Utils.print_log(
                f"File {save_path} => {os.path.getsize(save_path)} ")

            # 获取大小
            if os.path.getsize(save_path) == 0:
                if only_get_path:
                    return None
                os.remove(save_path)
                raise ValueError(f"Download failed: {url}")
            return save_path
        else:
            texts = url.split("?")[0].split("/")
            host = texts[2].replace(".", "_")
            base_model_path = os.path.abspath(os.path.join(
                Utils.get_models_path(), f"{host}_models"))

            file_name = texts[-1]
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1]
            md5_hash = Utils.Md5(url)

            save_path = os.path.join(
                base_model_path, f"{file_name_no_ext}.{md5_hash}{file_ext}")

            if os.path.exists(save_path) is False:
                if only_get_path:
                    return None
                os.makedirs(base_model_path, exist_ok=True)
                Utils.download_file(url, save_path)

            return save_path

    def print_log(*args):
        if os.environ.get("MZ_DEV", None) is not None:
            print(*args)

    def download_model(model_info, only_get_path=False):  
 
        url = model_info["url"]
        output = model_info["output"]
        save_path = os.path.abspath(
            os.path.join(Utils.get_models_path(), output))
        if not os.path.exists(save_path):
            if only_get_path:
                return None
            save_path = Utils.download_file(url, save_path)
        return save_path
    


    def load_lora(model, lora_path, strength_model):
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True) 
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return model_lora


modelscope_models_map = {
    "clay": {
        "clay.v1.pt": {
            "url": "https://modelscope.cn/api/v1/models/LLM-Research/Meta-Llama-3-8B-Instruct-GGUF/repo?Revision=master&FilePath=Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "output": "modelscope_models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "SHA256": "57b26bac2df51111affec600077708de06133b8f49e697723672657c7cbe3b9c",
        },
    },
}
