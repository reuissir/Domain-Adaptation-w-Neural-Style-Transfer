import os
import shutil
import random
import sys

KITTI_DIR = 'D:\DomainAdap.Neural\kitti\data_object_image_2'
image_dir = os.path.join(KITTI_DIR, 'training', 'image_2')
label_dir = os.path.join(KITTI_DIR, 'training', 'label_2')
val_dir =  os.path.join(KITTI_DIR, 'val')
split_ratio = 0.2

os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(val_dir, 'image_2'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'label_2'), exist_ok=True)

image_files = os.listdir(image_dir)
random.shuffle(image_files)

val_ratio = int(len(image_files) * split_ratio)

for i in range(val_ratio):
    image_file = image_files[i]
    label_file = image_file.replace('.png', '.txt')

    
    # source to destination path
    shutil.move(os.path.join(image_dir, image_file), 
                os.path.join(val_dir, 'image_2', image_file))
    shutil.move(os.path.join(label_dir, label_file),
                os.path.join(val_dir, 'label_2', label_file))
    
    # Display progress
    progress = (i + 1) / val_ratio * 100
    sys.stdout.write(f"\rProgress: {progress:.2f}%")
    sys.stdout.flush()

print(f"{val_ratio}개의 사진과 라벨 전송 끝!")