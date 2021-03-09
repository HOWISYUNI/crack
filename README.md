# YOLO v5를 활용한 crack detection
---
# 디렉터리 및 파일 설명

## datset/

    1. 학습 및 테스트르 위한 이미지 데이터로 구성.
    2. 반드시 이미지 폴더와 같은 준위에 labels 폴더가 존재해야 학습시킬 수 있다.
    3. 아래 설명할 디렉터리의 하위 구조는 대부분 images, labels 로 구성되며, 각각 이미지 데이터와 bounding box(이하 bbox) 정보가 저장돼있다.
    4. 디렉터리 내에 데이터 경로가 저장된 .txt가 있는 경우 .cache가 함께 존재한다.
        * .txt : train, test, validaion을 위한 데이터 경로 저장
        * .cache : 학습 시 생성되는 캐시파일
    5. 학습을 위한 데이터 배치 방법은 다양하지만 yaml 파일을 사용하는것이 직관적이며 확장성이 높다.
        1. 이미지, 라벨 데이터를 디렉터리로 묶되, 두 디렉터리가 같은 준위에 놓이게한다.
        2. 목적(train, train, test)에 따라 이미지 경로를 .txt 파일에 분리하여 기록한다.
        3. .yaml 파일에 데이터 경로가 저장된 .txt 파일 경로를 맵핑한다.
        4. train시 --data 인자의 정보로서 .yaml파일이 활용된다.

### dataset/coco128/
: coco128 데이터셋의 이미지 128장

### dataset/pure_carck/
1. https://data.mendeley.com/datasets/5y9wdsg2zt/2 에서 다운받은 crack 이미지
2. 위 링크 배포 데이터 중 Positive 디렉터리명은 images로 변경했으며, 이에 대한 labeling 정보는 labels에 저장돼있다.
3. Negative에는 crack이 없는 이미지로 구성돼있다.
4. `split_info`의 하위 디렉터리에는 목적에 맞게 이미지를 분리해 놓은 텍스트파일이 존재한다.

### dataset/SelectedCracks/
1. 선별된 crack이미지 65장이 들어있습니다. mendeley 데이터셋에서 연구자 본인이 특징적으로 생각하는 7가지 기준으로 데이터를 선별했다.
2. 선별 기준
    1. 가로
    2. 세로
    3. 구석
    4. 2줄로 나뉜 crack
    5. 회색 blob이 존재하는 경우
    6. crack이 비교적 두꺼운 경우
    7. 분기를 만드는 경우
3. coco128 데이터와 함께 학습시키기 위해 coco 데이터셋에 존재하지 않는 클래스 번호인 80번을 crack에 부여했다.

### dataset/test_images/
1. crack 이미지로만 학습시킨 모델의 분류 정확도 테스트를 위한 데이터
2. 테스트 스크립트 및 테스트 결과는 `../test_results/`에 저장된다.

### dataset/UnseenData/
1. coco + crack 으로 학습시킨 모델의 다중 클래스 객체 탐지를 위한 데이터
2. 학습에 참여하지 않았고, 인간 판단하에 같은 클래스에 해당하는 이미지에 대해 detection 테스트


## test_results/
1. crack으로만 학습시킨 모델의 분류 정확도 테스트결과
2. `dataset/test_images/`의 이미지를 이용한다.
3. 하위 디렉터리의 crack에는 detect 하지못한 crack이미지, noncrack에는 detect한 crack 이미지 저장

### 25/
: 학습된 모델에 `confidence threshold=.25`제한 시 결과

### 75/
: 학습된 모델에 `confidence threshold=.75`제한 시 결과

### test crack and non crack.ipynb
: 각 threshold별 분류 결과에 대한 시각화. confution matrix 정보 존재.


## .ipynb 파일
1. predict.ipynb : jupyter 파일로 터미널 동작
2. revise yaml.ipynb : python의 yaml 라이브러리를 활용해 .yaml 파일 수정. notebook이 아닌 직접 수정해도 동작은 동일하다.
3. split dataset.ipynb : `dataset`디렉터리에 존재하는 데이터를 목적에 맞게 .txt파일로 만들어낸다.

## .py 파일
: 아래 파일들은 crack이미지의 자동 라벨링을 위한 이미지 프로세싱 코드이다. mophology 연산과 otsu 알고리즘을 사용했다.
1. box.py : `cv2.recteangle`을 활용해 bbox를 그림
2. contour.py : `cv2.findContours`를 활용해 crack의 윤곽을 확인하여 bbox를 그림
3. lab.py : 이미지 프로세싱 알고리즘, 절차를 실험해보기위한 실험실
4. test.py : labeling 결과 좌표를 이미지에 다시 재현해 실제 이미지에 정상작동하는지 확인


# train, inference를 위한 간단한 터미널 명령어
아래 예시는 간단한 학습, 검출을 위한 명령어이며, 더 풍부한 기능을 위해 사용자가 원하는 파라미터를 변경 및 설정할 수 있다.

1. 학습
```
python train.py --img 416 --batch 16 --epochs 50 --data ../dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name withcoco_yolov5s_results
```

2. 이미지 단위로 inference
```
python detect.py --weights ./runs/train/crack_yolov5s_test_results/weights/best.pt --source ../dataset/UnseenData/7071-82.jpg --img 416 --save-txt --save-conf
```

3. 폴더 단위로 inference
```
python detect.py --weights ./runs/train/crack_yolov5s_test_results/weights/best.pt --source ../dataset/test_images/crack --img 416 --save-txt --save-conf --name crack
```
