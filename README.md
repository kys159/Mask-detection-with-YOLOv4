# :mask: Mask detection with Yolov4 :mask:

### 동국대학교 딥러닝의활용 최종 프로젝트
&nbsp;&nbsp; 동국대학교 딥러닝의활용 수업 최종 프로젝트로 딥러닝을 활용하여 실생활에 적용시켜본다. <br>
최근 Real-time Object Detection에서 높은 FPS와 좋은 성능을 보이는 __Yolo(You Only Look Once)v4__ 를 활용해, <br>
코로나19 예방을 위한 __마스크 착용 판별 AI__ 를 구축한다.

<br>

## :bulb: 전체적인 분석 목표
 - **Yolo v4**
   + YOLO(You Only Look Once)는 이미지 내의 bounding box와 class probability를 single regression problem으로 간주하여, 이미지를 한 번 보는 것으로 object의 종류와 위치를 추측한다. single convolutional network를 통해 multiple bounding box에 대한 class probablility를 계산하는 방식이다. <br>
 - 장점:
   + one-stage detector로 two-stage detector에 비해 속도가 매우 빠르다. 또한 기존의 다른 real-time detection system들과 비교할 때,2배 정도 높은 mAP를 보인다.
   + Image 전체를 한 번에 바라보는 방식으로 class에 대한 맥락적 이해도가 높다. 이로인해 낮은 backgound error(False-Positive)를 보인다.
   + Object에 대한 좀 더 일반화된 특징을 학습한다. 가령 natural image로 학습하고 이를 artwork에 테스트 했을때, 다른 Detection System들에 비해 훨씬 높은 성능을 보여준다.
 - 단점:
   + 상대적으로 낮은 정확도 (특히, 작은 object에 대해)
 - 여기서 Yolo의 단점인 작은 object에 대해서 잘 탐지해내지 못한다는 부분을 보완하여 cctv같은 작고 많은 사람이 나타나는 영상에서 마스크를 쓰지 않은 사람을 탐지해 낼 수 있는 알고리즘을 학습시키는 것이 목표이다.

<br>

## :bulb: Environment & Data setting

### 0. 실행환경 & 실험환경
- 실행환경
  + Linux 
  + CMake >= 3.12
  + CUDA >= 10.0
  + OpenCV >= 2.4
  + cuDNN >= 7.0
- 실험환경
  + Ubuntu 18.04
  + RAM 64GB
  + GPU 1070ti * 2
  + CUDA 10.1
  + OpenCV 4.42
  + cuDNN 7.64

<br>

### 1. YOLOv4 설치
* AlexAB의 github에 들어가서 git을 clone 해오거나 zip파일로 다운로드 받는다.<br>
```
git clone https://github.com/AlexeyAB/darknet.git
```

* darknet을 컴퓨터에 make하기 위해 makefile을 수정한다. GPU를 사용할 것이라면 GPU, CUDNN을 1로 설정하며 CUDNN_HALF는 학습속도를 향상시키는 부분이라고 한다. Open cv를 사용할 경우 OPENCV=1로 수정하고 LIBSO는 추후 응용 프로그램에 사용할 so 파일을 생성하는 옵션이다.<br>

```
GPU=1 
CUDNN=1 
CUDNN_HALF=1 
OPENCV=1 
AVX=0 
OPENMP=0 
LIBSO=1 
```

* make를 통해 darknet을 make 한다.<br>

```
make
```
<br>

### 2. 데이터 수집
선행 연구(주소)에서 활용한 데이터셋 887개와 구글에서 __"mask people" , "korea mask people", "wear mask people"__ 이라고 검색하여 크롤링한 후 중복되거나 그림파일이거나 혹은 파일이 열리지 않는 경우를 제외하여 데이터셋 756개를 수집하였다. 코드는 google_mask_image_crawling.ipynb 이다.

<br>

### 3. cfg파일 수정
cfg/yolov4-custom.cfg 파일을 AlexeyAB github에서 제시하는 최적의 값으로 수정한다. <br>

```
[net]
batch=16
subdivisions=16
width=608
height=608
channels=3
momentum=0.949
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = 4000
policy=steps
steps=3200,3600
scales=.1,.1
```

 + batch : 배치사이즈
 + subdivisions : 배치 사이즈를 얼마나 쪼개서 학습할 것인지에 대한 설정 값이다.
 + width 및 height : 기본값은 416이지만 608로 변경하여 학습 할 경우 해상도 향상으로 정확도가 좋아질 수 있으나 OOM ERROR가 발생할 수 있음.
 + max_batches : 언제까지 iteration을 돌건지 설정하는 값으로 본인 데이터의 클래스 갯수 * 2000을 제시한다.
 + steps : max_batches의 80%와 90%를 설정한다.

```
[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 0,1,2
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
scale_x_y = 1.2
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6
```
Ctrl+f를 이용하여 yolo부분을 검색해서 수정한다. yolo 부분에 classes와 convolutional 부분의 filters를 수정한다. classes는 훈련 시키는 데이터셋의 classes 갯수, filter 갯수는 (classes+5) * 3으로 AlexeyAB github에서 제시하고 있다. yolo 부분은 총 3개가 있어 모두 찾아 수정한다.

### 4. Data set 정의
##### 4.1 Padding
YOLO에는 어떠한 size의 데이터가 input되더라도 같은 size로 수정하여 model을 거치게 되므로 resize할 필요가 없다. 하지만 한정된 VRAM에서 일정한 size의 data를 input 함으로써 최적의 활용이 가능해지므로 데이터의 size를 608 * 608로 resize하였다. 코드는 ooo와 같다.

##### 4.2 데이터셋 경로 파일 수정
업로드 한 custom_data 형식에 맞게 txt 파일을 만든다.
```
├── custom_data/
   ├── dataset_label
   ├── obj.data
   ├── obj.names
   ├── test.txt
   └── train.txt
```
 - `dataset_label` 모든 데이터셋과 각 사진들의 label에대한 txt파일을 저장한다. AlexyAB의 Yolo_mark를 활용했으며 주소는 다음과 같다. https://github.com/AlexeyAB/Yolo_mark
 그 후, 앞서 2번에서 언급한 그림파일 혹은 라벨이 없는 이미지를 제거하였다. 코드는 OOO과 같다.
 - `obj.data` 전체적인 파일의 경로들을 저장한다.
 - `obj.names` classes name을 저장한다.
 - `test.txt` test에서 활용할 데이터 셋의 경로를 저장한다.
 - `train.txt` train에서 활용할 데이터 셋의 경로를 저장한다.

<br>

### 5. Weights
 #### Transfer learning을 위해 AlexeyAB github에서 제공하는 yolov4의 pretrain모델을 다운받는다.
 <a href="drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp">yolov4 weights from AlexeyAB github</a>
