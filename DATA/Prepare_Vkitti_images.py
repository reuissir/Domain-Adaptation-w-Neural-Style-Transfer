import os
import sys
import time
import shutil
import cv2

###
# The code first computs the total number of images
# After, the code accesses each image folders within the entire VKITTI image folder and extracts and copies each image to the target_image_path(line 60)"
# The images are also converted from .jpg to .pngs formats(through cv2) in the process
###

# Directories
ROOT_DIR = 'D:/DomainAdap.Neural/Data'
VKITTI_IMAGE_DIR = os.path.join(ROOT_DIR, 'VKitti', 'vkitti_2.0.3_rgb')
VKITTI_TRAIN_DIR = os.path.join(ROOT_DIR, 'VKITTI', 'VKITTI', 'train', 'images')

# image folders to be excluded
excluded = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']

resized_height = 640
resized_width = 640

start_time = time.time()


##
# Compute total number of images
##
total_images = 0
for scene in os.listdir(VKITTI_IMAGE_DIR):
    scene_path = os.path.join(VKITTI_IMAGE_DIR, scene)
    if os.path.isdir(scene_path):
        for subfolder in os.listdir(scene_path):
            if subfolder not in excluded:
                subfolder_path = os.path.join(scene_path, subfolder, 'frames', 'rgb', 'Camera_0')
                if os.path.isdir(subfolder_path):
                    total_images += len(os.listdir(subfolder_path))

processed_images = 0


##
# Access VKITTI folders and image files
# Resize and Copy VKITTI images to target_image_path
##
# for scene folders
for scene in os.listdir(VKITTI_IMAGE_DIR):
    scene_path = os.path.join(VKITTI_IMAGE_DIR, scene)

    if os.path.isdir(scene_path):
        for subfolder in os.listdir(scene_path):
            # for subfolders within each Scene folder
            if subfolder not in excluded:
                subfolder_path = os.path.join(scene_path, subfolder, 'frames', 'rgb', 'Camera_0')
                if os.path.isdir(subfolder_path):
                    print(f"Processing folder: {subfolder_path}")

                    for image_file in os.listdir(subfolder_path):
                        image_file_path = os.path.join(subfolder_path, image_file)

                        image_number = image_file.replace('rgb_', '').replace('.jpg', '')

                        # Rename and copy the image to the target directory
                        new_image_name = f"{scene}_{subfolder}_Camera0_{image_number}.png"  # jpg 대신 png 사용
                        target_image_path = os.path.join(VKITTI_TRAIN_DIR, new_image_name)

                        # 이미지를 opencv로 읽은 후 png로 저장
                        # OpenCV로 이미지 resize
                        img = cv2.imread(image_file_path)
                        resized_img = cv2.resize(img, (resized_width, resized_height))
                        cv2.imwrite(target_image_path, resized_img)

                        processed_images += 1
                        sys.stdout.write(f"\rProcessed: {processed_images}/{total_images} ({100 * processed_images/total_images:.2f}%)")
                        sys.stdout.flush()

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")