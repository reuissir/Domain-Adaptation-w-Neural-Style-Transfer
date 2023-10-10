import os

ROOT_DIR = 'D:/DomainAdap.Neural/Data/VKitti'
VKITTI_LABELS = os.path.join(ROOT_DIR, 'vkitti_2.0.3_textgt')
output_dir = os.path.join(ROOT_DIR, 'VKITTI', 'clone labels')
os.makedirs(output_dir, exist_ok=True)

orig_width = 1242
orig_height = 375

resized_width = 640
resized_height = 640

excluded = ['15-deg-left', '30-deg-left', '15-deg-right', '30-deg-right']

def VKITTI_label2KITTI_label(track_id):
    if track_id in [89, 90, 92]:
        return 1  # Convert to 'van' and then to 0
    elif track_id in [50, 87]:
        return 2  # Convert to 'truck' and then to 2
    else:
        return 0  # Convert to 'car' and then to 1


def convert_annotations(lines):
    yolo_annotations = {}
    
    for line in lines[1:]:
        parts = line.strip().split(' ')
        if len(parts) != 11:
            continue

        frame, camera_id, track_id, left, right, top, bottom, number_pixels, truncation_ratio, occupancy_ratio, is_moving = parts

        # original KITTI bounding box coordinates
        org_left = float(parts[3])
        org_right = float(parts[4])
        org_top = float(parts[5])
        org_bottom = float(parts[6])

        resized_left = org_left * resized_width / orig_width
        resized_right = org_right * resized_width / orig_width
        resized_top = org_top * resized_height / orig_height
        resized_bottom = org_bottom * resized_height / orig_height


        # convert bbox coordinates into x, y, w, h + normalization
        bbox_center_x = ((resized_left + resized_right) / 2.0 )/ resized_width
        bbox_center_y = ((resized_top + resized_bottom) / 2.0 )/ resized_height
        bbox_width = (resized_right - resized_left) / resized_width
        bbox_height = (resized_bottom - resized_top) / resized_height

        # Get VKITTI class label [trackID] and convert to YOLO class
        vkitti_class = VKITTI_label2KITTI_label(int(track_id))

        frame_camera_id_key = int(frame), int(camera_id)

        # Check if it's a new frame, and initialize the list if needed
        if frame_camera_id_key[1] == 0:
            # Ensure there's a list to append to
            if frame_camera_id_key not in yolo_annotations:
                yolo_annotations[frame_camera_id_key] = []

            # Create YOLO format annotation
            yolo_annotation = f"{vkitti_class} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n"
            
            # Append the annotation
            yolo_annotations[frame_camera_id_key].append(yolo_annotation)


    return yolo_annotations

# Iterate through each scene folder in VKITTI_LABELS
for scene_folder in os.listdir(VKITTI_LABELS):
    scene_folder_path = os.path.join(VKITTI_LABELS, scene_folder)
    if os.path.isdir(scene_folder_path):       
        # Iterate through subfolders within the scene folder

        for clone in os.listdir(scene_folder_path):
            if clone == 'clone':
                clone_path = os.path.join(scene_folder_path, clone)
            # Check if it's a directory and not excluded
                if os.path.isdir(clone_path):
                    bbox_file_path = os.path.join(clone_path, "bbox.txt")

                # Check if "bbox.txt" file exists
                    if os.path.isfile(bbox_file_path):
                        with open(bbox_file_path, 'r') as f:
                            lines = f.readlines()

                        # Convert annotations to YOLO format
                        yolo_annotations = convert_annotations(lines)

                        # Save YOLO format annotations to separate text files for each frame
                        for (frame, camera_id), annotations in yolo_annotations.items():
                            # Create a file name based on the frame
                            output_file_name = f"{scene_folder}_{clone}_Camera{camera_id:01d}_{frame:05d}.txt"
                            output_file_path = os.path.join(output_dir, output_file_name)

                            try:
                                with open(output_file_path, 'w') as f:
                                    f.writelines(annotations)
                                # print(f"Created {output_file_path}")
                            except Exception as e:
                                print(f"Error creating {output_file_path}: {e}")

print("Conversion complete.")