이름 | 역할
--- | ---
안희상 | 데이터 정제, Neural Style Transfer, YOLOv5 train
양원규 | Neural Style Transfer, YOLOv5 train, 결과 분석

# Domain-Adaptation-w-Neural-Style-Transfer
Bridging the gap between the source domain(VKITTI) and the target domain (BDD100k) through neural style transfer applied synthetic data.

## Introduction:
Virtual Kitti(VKITTI) offers synthetic object detection data, produced with Unity, that simulates real-world environments. It offers real-time drive images in overcast, morning, sunset, foggy, and rainy backgrounds that are absent in KITTI. On top of VKITTI, we sought to create a stylized dataset with neural style transfer which would perform better than VKITTI on a never-before-seen dataset(BDD100k). We trained two groups of datasets: 1: "KITTI + VKITTI", 2: "KITTI + VKITTI clone + NST VKITTI Clone" . Testing was done on BDD100k to compare the results. 

#### Problem at hand: 
- YOLOv5 trained on KITTI + VKITTI performs well on KITTI, however, suffers difficulty in detecting in environments not offered by VKITTI nor KITTI.

* Scenes where the model trained with KITTI + VKITTI showed difficulty in detection
  
![c93bd9ce-ea579ed8](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/1017445f-4a09-4534-8236-b2fdcc204192) ![c093f8be-262822a7](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/452879c5-b9cf-4611-9151-f800b859ebb0)
![c94ceb93-5b8608c4](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/882200e2-291e-4798-becd-93b9545e9175)

#### Cause:
- BDD100k provides images taken 24 hours whereas KITTI and VKITTI Clone were taken only during the day
- Difference in scenery(background/environment, lighting conditions)
- Fails to perform detection especially in cases when the image is taken during nighttime or when sunlight is strong

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
- Neural style transfer was conducted on VKITTI Clone.
  ** Hyperparameters:** content layer, style layer, content weight, style weight, total variation weight,            
                        init_method[style, content, random(gaussian or white noise)]
  
  We experimented on different style weights and content weights but our primary focus was on the convolutional layers which we would extract our feature representations.      In   the paper, they used conv4_2 because it worked best for their artistic style transfers. On the contrary, we concentrated on keeping the edges and overall shape of       the cars   alive throughout the transfer. Instead of using high-level layers, we chose low-level layers where low-level features are stored.

  **We discovered the existence of different style transfer types: artistic, realistic, cartoon, etc during the project. Yet, we preceded with our model with a hope that     
    maybe the model will still be forced to find the right features inspite of all the diversity in the data.** 
  
  - **Content layer:** conv_2 showed best results with content information still strong.
      - Differed per style image. Sometimes conv_3 worked better (ex. Fog)
  - **Style layer:** style information was extracted from every layer except the layer responsible for the content.
  - Different **init_methods** were used for various styles(fog-content, rain-style, night-content, sunset-content)
  - The content images were set as VKITTI Clone(2066 images)
  - A total of 2066 * 4(styles), 8264 images were stylized to form our NST Dataset.
  - KITTI + VKITTI clone + NST dataset was trained to be compared with the original KITTI + VKITTI dataset.

night | fog | rain | sunset
--- | --- | --- | --- |
![night_Scene01_clone_Camera0_00130 (1)](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/d35d948f-764f-4dca-8dfc-db7633349ea9)|![fog_Scene01_clone_Camera0_00112](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/e18fcdde-a3ab-4a19-8d80-f47a87d2816d)|![rain_Scene01_clone_Camera0_00308](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/1cfdf02a-a63f-4b78-936c-5d37d4db6473)|![sunset_Scene18_clone_Camera0_00222](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/bf3469f8-4662-48e2-88e4-89039dc39b76)

### Object detection: YOLOv5m6
- We tried YOLOv5m6 and YOLOl6, however, found that YOLOl6 took too long in our environment. YOLOv5m6 was the best model that matched our conditions.
- All training runs were run at 100 epochs, 16 batches, pretrained(COCO-128) weights offered by Ultralytics.
- We ran each dataset with different hyperparameters but found the default settings of Ultralytics worked best(Perhaps 100 epochs was not enough).


* In short, our experiment with domain adaptation through neural style transfer failed. 
* When trained with KITTI + VKITTI Clone + VKITTI Clone NST, it scored a mAP.50 of .820 during validation.
* When validated on BDD100k, the results were a terrible mAP.50 .142. (The need to proceed on this experiment could not be seen)
* Below are the results

## Training / Validation:

![image](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/66e94bbb-39c1-410d-8f1c-4a3f33112787)

 


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
  





  

