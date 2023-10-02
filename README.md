# Domain-Adaptation-w-Neural-Style-Transfer
Bridging the gap between reality and simulation through neural style transfer

#### Problem at hand:
YOLOv5 trained on KITTI + VKITTI performs well on KITTI dataset, however, fails to produce promising results when tested on BDD100k.

##### cause:
- Difference in scenery(background/environment, lighting conditions)
- Fails to perform detection especially in cases when the image is taken during nighttime or when sunlight is strong

![image](https://github.com/reuissir/Domain-Adaptation-w-Neural-Style-Transfer/assets/96709570/6458a84e-bd36-4e60-a749-bb1a41439062)


### 팀원 구성
- 안희상: Neural Style Transfer, 데이터 전처리
- 양원규: YOLOv5, Neural Style Transfer 

### Environment
- Ryzen 3600, 2070 Super 8gb, 16 ram
- Google Colab Pro+ (A100)

### 사용 모델
> YOLOv5 l6, Ultralytics
[https://github.com/ultralytics/yolov5]
> Pytorch-Neural-Style-Transfer
[https://github.com/reuissir/pytorch-neural-style-transfer]


