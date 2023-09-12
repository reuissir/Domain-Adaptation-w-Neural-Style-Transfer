import numpy as np
import cv2
import os
import sys

# Defining Paths and Variables
root_dir = "C:/Users/BrandAnn/yolov5"

## image dir & label dir
kitti_img_path ='C:/Users/BrandAnn/yolov5/kitti/data_object_image_2/training/image_2'
kitti_label_path = 'C:/Users/BrandAnn/yolov5/kitti/label/training/label_2'

## path where transformed labels will be saved
kitti_label_tosave_path = 'C:/Users/BrandAnn/yolov5/kitti/transformed'

index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX


# Reading Class Names

## Open and read the file kitti.names which contains class names for object detection
kitti_names = open('kitti.names','r')
kitti_names_contents = kitti_names.readlines()

# Load Image and Label Filenames

kitti_images = os.listdir(kitti_img_path)
kitti_labels = os.listdir(kitti_label_path)

kitti_images.sort()
kitti_labels.sort()


# Create Class-to_Index Dictionary

kitti_names_dic_key = ["car", "van", "truck", "DontCare"]

# Create the class-to-index dictionary
values = range(len(kitti_names_dic_key))
kitti_names_num = dict(zip(kitti_names_dic_key, values))
print(kitti_names_num)

# Create a new file called train.txt

## loop over image filenames + construct the full path + 'img' filename
f = open('train.txt','w')
for img in kitti_images:
    ## write to the train.txt file
    f.write(kitti_img_path+'/'+img+'\n')
f.close()

# Process Image and Labels

## for each image filename in kitti_images
for indexi in range(len(kitti_images)):
    ## concatenate path
    transformed_label_path = os.path.join(kitti_label_tosave_path, kitti_labels[indexi])    
    kitti_img_totest_path = os.path.join(kitti_img_path, kitti_images[indexi])
    kitti_label_totest_path = os.path.join(kitti_label_path, kitti_labels[indexi])
    
    kitti_img_totest = cv2.imread(kitti_img_totest_path)

    img_height, img_width = kitti_img_totest.shape[0], kitti_img_totest.shape[1]
    
    kitti_label_totest = open(kitti_label_totest_path, 'r')
    label_contents = kitti_label_totest.readlines()

    real_label = open(transformed_label_path, 'w')
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

            intx1 = int(x1)
            inty1 = int(y1)
            intx2 = int(x2)
            inty2 = int(y2)

            bbox_center_x = float((x1 + (x2 - x1) / 2.0) / img_width)
            bbox_center_y = float((y1 + (y2 - y1) / 2.0) / img_height)
            bbox_width = float((x2 - x1) / img_width)
            bbox_height = float((y2 - y1) / img_height)

            # Construct the line with class, bbox, truncation, and occlusion
            line_to_write = (
                    f"{kitti_names_num[class_str]} "
                    f"{bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height} "
                    f"{truncation} {occlusion}\n"
                )
                
            real_label.write(line_to_write)
            sys.stdout.write(str(int((indexi / len(kitti_images)) * 100)) + '% ' + '*******************->' "\r")
            sys.stdout.flush()
      

    #cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)    
    #cv2.waitKey()
real_label.close()
kitti_names.close()
print("Labels tranform finished!")
