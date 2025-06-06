import os
import numpy as np
import SimpleITK as sitk
import time
from skimage import morphology


def download():
    print("你将要下载HaN-Seg数据集。在下载前，你确认已通过https://han-seg2023.grand-challenge.org链接查看数据集的使用条款，并同意遵守这些条款。你认可下载该数据集引起的任何法律责任均由你本人承担。")

    print("You are about to download the HaN-Seg dataset. Before downloading, you confirm that you have reviewed the terms of use for the dataset at https://han-seg2023.grand-challenge.org and agree to comply with these terms. You acknowledge that any legal responsibility arising from downloading this dataset is solely yours.")
    
    name = input("在此签名 Please sign here: ")
    if not name:
        print("未输入签名，下载将被取消。")
        return
    print(f"感谢你的签名：{name}。现在开始下载HaN-Seg数据集...")
    print("Downloading HaN-Seg dataset...")
    
    path = 'https://zenodo.org/records/7442914/files/HaN-Seg.zip?download=1'
    os.system(f'wget -O HaN-Seg.zip {path}')
    os.system('unzip HaN-Seg.zip -d HaN-Seg')
    print("下载完成。数据集已解压到HaN-Seg目录。请确保遵守数据集的使用条款。")
    return name, time.time()


def process_dataset():
    print("正在处理HaN-Seg数据集...")
    dataset_path = 'HaN-Seg/HaN-Seg/set_1'
    out_dir = 'HaN-Seg'
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"数据集目录 '{dataset_path}' 不存在，请先下载数据集。")
        return
    
    files = os.listdir(dataset_path)
    for file in files:
        if "case" not in file:
            continue
        image = file + '_IMG_CT.nrrd'
        seg_1 = file + '_OAR_A_Carotid_L.seg.nrrd'
        seg_2 = file + '_OAR_A_Carotid_R.seg.nrrd'
        
        img_numpy = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dataset_path, file, image)))
        seg_1_numpy = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dataset_path, file, seg_1)))
        seg_2_numpy = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dataset_path, file, seg_2)))
        
        
        seg = np.zeros_like(img_numpy, dtype=np.uint8)
        seg[seg_1_numpy > 0] = 1
        seg[seg_2_numpy > 0] = 2
        
        
        cline = morphology.skeletonize_3d(seg > 0).astype(np.uint8)
        cline_map = seg.copy()
        cline_map[cline == 0] = 0

        np.savez_compressed(os.path.join(out_dir, file + '.npz'), img=img_numpy, seg=seg, cline=cline_map)
    return 

if __name__ == "__main__":
    name, timestamp = download()
    if name and timestamp:
        print(f"下载者: {name}, 时间戳: {timestamp}")
        process_dataset()
    else:
        print("下载未完成或取消。")