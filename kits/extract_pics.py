import os
from PIL import Image
import numpy as np
from tqdm import tqdm
extract_source_1 = "C:/Users/WhuLi/Documents/FUN/raw/"
extract_source_2 = "C:/Users/WhuLi/Documents/FUN/face/"

extract_target_path = "C:/Users/WhuLi/Documents/FUN/pix2pix/"

shape = [256, 256]


def scan_file(file_dir, max_num=None):
    if max_num:
        count = 0
        files = []
        for roo, dirs, file in os.walk(file_dir):
            files.append(file)
            count += 1
            if count >= max_num:
                break
        return files[0][0:max_num]
    else:
        files = []
        for roo, dirs, file in os.walk(file_dir):
            files.append(file)
        return files[0]


def clean_pix(imga):
    img_array = np.array(imga)
    shape = img_array.shape
    height = shape[0]
    width = shape[1]
    dst = np.zeros((height, width, 3))
    for h in range(0, height):
        for w in range(0, width):
            (b, g, r) = img_array[h, w]
            if np.linalg.norm(np.array([b, g, r])-np.array([255, 255, 255])) < 120:
                pass
            elif np.linalg.norm(np.array([b, g, r])-np.array([0, 0, 0])) < 120:
                pass
            else:
                img_array[h, w] = (255, 0, 0)
            dst[h, w] = img_array[h, w]
    return Image.fromarray(np.uint8(dst))


def resize_merge(imga, imgb, shape):

    joint = Image.new('RGB', (shape[0]+shape[0], shape[1]))
    loc1, loc2 = (0, 0), (shape[0], 0)
    joint.paste(imga, loc1)
    joint.paste(imgb, loc2)
    return joint


if __name__ == "__main__":
    files_A = scan_file(extract_source_1)
    for f in tqdm(files_A):
        try:
            imga = Image.open(extract_source_1 +
                              f).resize(shape, Image.ANTIALIAS)
            # imga = clean_pix(imga)
            imgb = Image.open(extract_source_2 +
                              f).resize(shape, Image.ANTIALIAS)
            img = resize_merge(imga, imgb, shape)
            img.save(extract_target_path+f)
        except Exception:
            pass
