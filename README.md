# Domain-Adaptation-w-Neural-Style-Transfer
Bridging the gap between the source domain(VKITTI) and the target domain (BDD100k) through neural style transfer applied synthetic data

#### Problem at hand:
- Expensive cost of producing various datasets. 
- YOLOv5 trained on KITTI + VKITTI performs well on KITTI dataset, however, fails to produce promising results when tested on BDD100k.

Trained| Test | mAP.50|
--- | --- | --- |
KITTI + VKITTI| BDD100k | .342|


#### Cause:
- BDD100k provides images taken 24 hours whereas KITTI and VKITTI Clone were taken during the day
- Difference in scenery(background/environment, lighting conditions)
- Fails to perform detection especially in cases when the image is taken during nighttime or when sunlight is strong

![image](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/6458a84e-bd36-4e60-a749-bb1a41439062)

### **Method**:
- Train model on synthetic data stylized with different background scenes(night, rain, fog, sunset) to adapt to the environment variables prevalent in the target domain.
  --> Compare performance of VKITTI + KITTI trained model vs. VKITTI + VKITTI Clone + VKITTI Clone NST(our dataset)         on BDD100k  

#### Models used
> YOLOv5m6, Ultralytics
[https://github.com/ultralytics/yolov5]

> Pytorch-Neural-Style-Transfer
[https://github.com/reuissir/pytorch-neural-style-transfer]

**Note:** We modified the code of Neural Style Transfer to perform sequential neural style transfer on an entire image directory.
During experiments: we used the main code to produce and evaluate a single output at a time

#### Environment
- Ryzen 3600, 2070 Super 8gb, 16 ram
- Google Colab Pro+ (A100)

We produced most of our dataset through Google Colab while most of our training was done on my local environment.
We experimented with different augmentation techniques and two optimizers[SGD, AdamW] to bring out the best performance during train time.

### Neural Style Transfer
- Neural style transfer was conducted on the VKITTI Clone data directory.
  ** Hyperparameters:** content layer, style layer, content weight, style weight, total variation weight,             
                        init_method[style, content, random(gaussian or white noise)]
  - Content layer conv_4_2 showed best results with content information still strong.
  - Only when stylizing night did we use conv_2.
  - Style informations were extracted from every layer except the layer responsible for the content.
  - Different init_methods were used for various styles(fog-content, rain-style, night-content, sunset-content)
  - The content images were set as VKITTI Clone(2066 images)
  - A total of 2066 * 4(styles), 8264 images were stylized to form our NST Dataset.
  - KITTI + VKITTI clone + NST dataset was trained to be compared with the original KITTI + VKITTI dataset.


    
### Object detection: YOLOv5m6
- We tried YOLOv5m6 and YOLOl6, however, found that YOLOl6 took too long in our environment. YOLOv5m6 was the best      model that matched our environment.
- All training runs were run at 100 epochs, 16 batches, pretrained(COCO-128) weights offered by Ultralytics.




## Data Preprocess

#### Labels to YOLO format
YOLO Format
![image](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/5f03cad8-6326-4a75-8828-2efedd9e70fa)

1. Unite all classes under {0: 'Car', 1: 'Van', 2: 'Truck, 3: 'DontCare'}
   - return int value
3. Adjust bounding box coordinates(left, right, top bottom) to resized image
4. Convert bounding box coordinates to x, y, w, h
5. Normalize bounding box coordinates

The DATA folder in this repo contains different codes for processing the data.
Prepare_BDD100k.py (prepare images and data)
Prepare_Kitti.py (prepare images and data)

* VKITTI
  * **Note**: VKITTI has a different directory structure where images are separated in different folders(Scene01~Scene20)
      - Two codes are provided: one for annotations, the other for images
Prepare_VKitti_labels.py
  - Creates one label per image based on [frame][camera_id]
  - Image + labels from folder [camera01, left-15-degrees, left-30-degrees, right-15-degrees, right-30-degrees] were       excluded
Prepare_VKitti_images.py
  - Code to accumulate images contained in separate Scene folders into one directory
  - Rename files according to 

#### etc.
- resize_images.py -- > resize images to 640*640
- Draw_BBox --> draws bounding box coordinates from annotation file(useful to check if labels were converted properly)
- 





  

