import os
import sys
import time
import shutil

ROOT_DIR = 'D:/DomainAdap.Neural/VKitti'
VKITTI_IMAGE_DIR = os.path.join(ROOT_DIR, 'vkitti_2.0.3_rgb')
VKITTI_TRAIN_DIR = os.path.join(ROOT_DIR, 'VKITTI', 'train', 'images')
VKITTI_LABEL_DIR = os.path.join(ROOT_DIR, 'VKITTI', 'train', 'labels')

excluded = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']

start_time = time.time()

# Compute total number of images
total_images = 0
for scene in os.listdir(VKITTI_IMAGE_DIR):
    scene_path = os.path.join(VKITTI_IMAGE_DIR, scene)
    if os.path.isdir(scene_path) and scene not in excluded:
        for subfolder in os.listdir(scene_path):
            subfolder_path = os.path.join(scene_path, subfolder, 'frames', 'rgb')
            if os.path.isdir(subfolder_path) and subfolder not in excluded:
                for camera in os.listdir(subfolder_path):
                    camera_path = os.path.join(subfolder_path, camera)
                    if os.path.isdir(camera_path):
                        total_images += len(os.listdir(camera_path))

processed_images = 0

# for scene folders
for scene in os.listdir(VKITTI_IMAGE_DIR):
    scene_path = os.path.join(VKITTI_IMAGE_DIR, scene)
    
    # for subfolders in scene folders
    if os.path.isdir(scene_path) and scene not in excluded:
        
        for subfolder in os.listdir(scene_path):
            subfolder_path = os.path.join(scene_path, subfolder, 'frames', 'rgb')
            
            if os.path.isdir(subfolder_path) and subfolder not in excluded:
                print(f"Processing subfolder: {subfolder_path}")
            
                
                for camera in os.listdir(subfolder_path):
                    camera_path = os.path.join(subfolder_path, camera)
                    
                    if os.path.isdir(camera_path):
                        print("Path is correct")
                    
                        for image_file in os.listdir(camera_path):
                            image_file_path = os.path.join(camera_path, image_file)

                            camera_number = f"camera{int(camera.split('_')[-1]):02}"  # converts camera number to two digits
                            image_number = image_file.replace('rgb_', '').replace('.jpg', '')

                            # Rename and copy the image to the target directory
                            new_image_name = f"{scene}_{subfolder}_{camera_number}_{image_number}.jpg"
                            target_image_path = os.path.join(VKITTI_TRAIN_DIR, new_image_name)
                            shutil.copy(image_file_path, target_image_path)  

                            processed_images += 1
                            sys.stdout.write(f"\rProcessed: {processed_images}/{total_images} ({100 * processed_images/total_images:.2f}%)")
                            sys.stdout.flush()

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")