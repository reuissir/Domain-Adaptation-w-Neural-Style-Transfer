import os

ROOT_DIR = 'D:\DomainAdap.Neural\VKitti'
VKITTI_LABELS = os.path.join(ROOT_DIR, 'vkitti_2.0.3_textgt')
output_dir = os.path.join(ROOT_DIR, 'VKITTI_ANNOTATIONS')
os.makedirs(output_dir, exist_ok=True)

image_width = 1242
image_height = 375

excluded = ['15-deg-left', '30-deg-left', '15-deg-right', '30-deg-right']

def VKITTI_label2KITTI_label(track_id):
    if track_id in [89, 90, 92]:
        return 2  # Convert to 'van' and then to 0
    elif track_id in [50, 87]:
        return 3  # Convert to 'truck' and then to 2
    else:
        return 1  # Convert to 'car' and then to 1

#vkitti_class_to_yolo = {
#    'van': 2,
#    'car': 1,
#    'truck': 3,
#}

def convert_annotations(lines):
    yolo_annotations = {}
    
    for line in lines[1:]:
        parts = line.strip().split(' ')

        if len(parts) != 11:
            continue

        frame, camera_id, track_id, left, right, top, bottom, number_pixels, truncation_ratio, occupancy_ratio, is_moving = parts

        # YOLO annotations
        left = float(parts[3])
        right = float(parts[4])
        top = float(parts[5])
        bottom = float(parts[6])
        truncation_ratio = float(parts[8])
        occupancy_ratio = float(parts[9])

       # convert bbox coordinates into x, y, w, h + normalization
        bbox_center_x = ((left + right) / 2.0 )/ image_width
        bbox_center_y = ((top + bottom) / 2.0 )/ image_height
        bbox_width = (right - left) / image_width
        bbox_height = (bottom - top) / image_height

        # Get VKITTI class label [trackID] and convert to YOLO class
        vkitti_class = VKITTI_label2KITTI_label(int(track_id))

        frame_camera_id_key = int(frame), int(camera_id)

        # Check if it's a new frame, and initialize the list if needed
        if frame_camera_id_key not in yolo_annotations:
            yolo_annotations[frame_camera_id_key] = []

        # Create YOLO format annotation with truncation and occlusion
        yolo_annotation = f"{vkitti_class} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height} {truncation_ratio} {occupancy_ratio}\n"
        yolo_annotations[frame_camera_id_key].append(yolo_annotation)

    return yolo_annotations

# Iterate through each scene folder in VKITTI_LABELS
for scene_folder in os.listdir(VKITTI_LABELS):
    scene_folder_path = os.path.join(VKITTI_LABELS, scene_folder)

    # Check if it's a directory and not excluded
    if os.path.isdir(scene_folder_path) and scene_folder not in excluded:
        
        # Iterate through subfolders within the scene folder
        for subfolder in os.listdir(scene_folder_path):
            subfolder_path = os.path.join(scene_folder_path, subfolder)

            # Check if it's a directory and not excluded
            if os.path.isdir(subfolder_path) and subfolder not in excluded:
                bbox_file_path = os.path.join(subfolder_path, "bbox.txt")

                # Check if "bbox.txt" file exists
                if os.path.isfile(bbox_file_path):
                    with open(bbox_file_path, 'r') as f:
                        lines = f.readlines()

                    # Convert annotations to YOLO format
                    yolo_annotations = convert_annotations(lines)

                    # Save YOLO format annotations to separate text files for each frame
                    for (frame, camera_id), annotations in yolo_annotations.items():
                        # Create a file name based on the frame
                        output_file_name = f"{scene_folder}_{subfolder}_camera{camera_id:02d}_{frame:05d}.txt"
                        output_file_path = os.path.join(output_dir, output_file_name)

                        try:
                            with open(output_file_path, 'w') as f:
                                f.writelines(annotations)
                            print(f"Created {output_file_path}")
                        except Exception as e:
                            print(f"Error creating {output_file_path}: {e}")

print("Conversion complete.")