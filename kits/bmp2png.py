import os
from PIL import Image
from tqdm import tqdm

source_path = "C:/Users/WhuLi/Documents/HRSC2016/FullDataSet/AllImages/"
target_path = "C:/Users/WhuLi/Documents/HRSC2016/FullDataSet/AllImages_png/"


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


if __name__ == "__main__":
    source_files = scan_file(source_path)
    for s in tqdm(source_files):
        img = Image.open(source_path+s)
        name = s[0:-4]
        img.save(target_path+name+".png")
