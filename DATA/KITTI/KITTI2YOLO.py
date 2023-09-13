import numpy as np
import cv2
import os
import sys

# Defining Paths and Variables
root_dir = "D:/DomainAdap.Neural/kitti"

## image dir & label dir
kitti_img_path ='D:/DomainAdap.Neural/kitti/data_object_image_2/training/image_2'
kitti_label_path = 'D:/DomainAdap.Neural/kitti/label/training/label_2'

## path where transformed labels will be saved
converted_path = 'C:/Users/BrandAnn/yolov5/kitti/transformed'

index = 0

## Read kitti.names which contains class names of KITTI
with open('kitti.names','r') as file:
    kitti_names = file.readlines()

# Load Image and Label Filenames
kitti_images = os.listdir(kitti_img_path)
kitti_labels = os.listdir(kitti_label_path)

kitti_images.sort()
kitti_labels.sort()


# Create Class-to_Index Dictionary
kitti_names_dic_key = ["car", "van", "truck", "DontCare"]
values = range(len(kitti_names_dic_key))

kitti_class_index = dict(zip(kitti_names_dic_key, values))
print(kitti_class_index)

# Create a new file called train.txt
## loop over image filenames + construct the full path + 'img' filename
f = open('train.txt','w')
for img in kitti_images:
    ## write to the train.txt file
    f.write(kitti_img_path+'/'+img+'\n')
f.close()

# Process Image and Labels
for indexi in range(len(kitti_images)):
    ## concatenate path
    transformed_label_path = os.path.join(converted_path, kitti_labels[indexi])    
    kitti_images_ = os.path.join(kitti_img_path, kitti_images[indexi])
    kitti_labels_ = os.path.join(kitti_label_path, kitti_labels[indexi])
    
    kitti_img_ = cv2.imread(kitti_images_)

    img_height, img_width = kitti_img_.shape[0], kitti_img_.shape[1]
    
    kitti_label_contents = open(kitti_labels_, 'r')
    label_contents = kitti_label_contents.readlines()

    final_label = open(transformed_label_path, 'w')

    # Access label contents in order to convert them to YOLO format
    for line in label_contents:
        data = line.split(' ')
        x = y = w = h = 0
        if len(data) == 15:
            class_str = data[0]
            if class_str not in ["car", "van", "truck"]:
                class_str = 'DontCare'
            # Parse truncation and occlusion values from the label
            truncation = float(data[1])
            occlusion = float(data[2])

            x1 = float(data[4])
            y1 = float(data[5])
            x2 = float(data[6])
            y2 = float(data[7])

            # convert bbox coordinates into x, y, w, h + normalization
            bbox_center_x = (x1 + (x2 - x1) / 2.0) / img_width
            bbox_center_y = (y1 + (y2 - y1) / 2.0) / img_height
            bbox_width = (x2 - x1) / img_width
            bbox_height = (y2 - y1) / img_height

            # Write label contents into YOLO format
            line_to_write = (
                    f"{kitti_class_index[class_str]} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height} {truncation} {occlusion}\n"
                )
                
            final_label.write(line_to_write)
            sys.stdout.write(str(int((indexi / len(kitti_images)) * 100)) + '% ' + '*******************->' "\r")
            sys.stdout.flush()
      

    #cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)    
    #cv2.waitKey()
final_label.close()
kitti_names.close()
print("Label tranformations finished!")
