# Domain-Adaptation-w-Neural-Style-Transfer
Bridging the gap between reality and simulation through neural style transfer. Domain adaptation of the source domain(VKITTI) to (BDD100k).

#### Problem at hand:
- Expensive cost of producing various datasets. 
- YOLOv5 trained on KITTI + VKITTI performs well on KITTI dataset, however, fails to produce promising results when     tested on BDD100k.

#### Cause:
- Difference in scenery(background/environment, lighting conditions)
- Fails to perform detection especially in cases when the image is taken during nighttime or when sunlight is strong

![image](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/6458a84e-bd36-4e60-a749-bb1a41439062)

#### Methods:
- Train model on synthetic data stylized with different background scenes(night, rain, fog, sunset) to adapt to the environment variables prevalent in the target domain.
  --> Compare performance of VKITTI + KITTI trained model vs. VKITTI + VKITTI Clone + VKITTI Clone NST(our dataset)         on BDD100k  
- Use BDD100k as the validation dataset during training

### Models used
> YOLOv5m6, Ultralytics
[https://github.com/ultralytics/yolov5]

> Pytorch-Neural-Style-Transfer
[https://github.com/reuissir/pytorch-neural-style-transfer]

**Note:** We modified the code of Neural Style Transfer to perform sequential neural style transfer on an entire image directory.
During experiments: we used the main code to produce and evaluate a single output at a time

### Environment
- Ryzen 3600, 2070 Super 8gb, 16 ram
- Google Colab Pro+ (A100)

We produced most of our dataset through Google Colab while most of our training was done on my local environment.
We experimented with different augmentation techniques and two optimizers[SGD, AdamW] to bring out the best performance during train time.



