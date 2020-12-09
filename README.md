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

### 1. YOLOv4 설치
* AlexAB의 github에 들어가서 git을 clone 해오거나 zip파일로 다운로드 받는다.<br>
```
git clone https://github.com/AlexeyAB/darknet.git
```
<br>
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

<br>
* make를 통해 darknet을 make 한다.<br>
```
make
```

### 2. 데이터 수집
선행 연구에서 활용한 데이터셋 ooo개와 구글에서 ooo라고 검색하여 크롤링한 데이터셋 ooo개를 수집하였다.

### 2. cfg파일 수정









### 1. Weights
 #### Transfer learning을 위해 AlexeyAB github에서 제공하는 yolov4의 pretrain모델을 다운받는다.
