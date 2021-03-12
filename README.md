# Crack Detection using YOLOv5 

## Feature
1. YOLO v5를 활용하여 이미지 내 __crack을 검출__
2. coco dataset과 함께 학습시켜 __crack을 포함한 다중 객체 검출__
2. 전통적인 image processing 기법 중 **otsu 알고리즘과 morphology 연산**을 활용__해 **자동 labelling**을 수행

## 학습을 위한 파일 설정
1. __(데이터셋) -- (`.txt` 맵핑) -- (`.yaml` 설정) -- (yolov5)__
    * yolov5의 `train.py`의 `--data` 파라미터의 값으로 `.yaml`설정파일을 지정해, 데이터셋의 위치를 담은 `.txt`맵핑파일로 연결.
    * `.txt`파일은 목적에 따라 분리되며, 목적에 부합하는 이미지 경로가 저장돼있다.
2. 데이터셋 : 학습 이미지와 bounding box(이하 bbox)정보가 필요합니다.
    * 본 프로젝트에서는 학습 이미지를 `images/`, bbox 정보는 `labels/`에 모아뒀습니다.
    * `images/`와 `labels/` 디렉터리는 동일 준위에 위치해야하며, 디렉터리명은 변경가능합니다.
    * `images/`
        * `.jpg` 확장자의 이미지 파일을 사용합니다.
        * 이미지 출처 : [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2) 
    * `labels/`
        * crack을 검출하는 bbox의 좌표 정보에 대한 `.txt` 파일
        * 자세한 내용 : [ultralytics/yolov5 - Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
3. `.txt` 맵핑 파일
    * train, test, validation의 모든 대상 이미지 파일 경로가 저장된 파일.
    * 목적에 따라 파일을 분리한다.
4. `.yaml` 설정파일
    * `.txt`맵핑파일과 클래스명, 클래스 갯수를 담고있다.
    * `.yaml`의 `names`리스트 원소는 클래스 번호와 맵핑된다.
    * 첫 번째 원소는 0부터 시작하여 1씩증가한다.

## 디렉터리 및 파일 설명
1. `dataset/` : 학습, 테스트를 위한 이미지로 구성.
    1. `coco128/` : coco128 데이터셋의 이미지 128장
    2. `pure_crack/` : 원본 crack 이미지 데이터셋
        1. `images/` : crack 이미지 20000장
        2. `labels/` : crack bbox 정보
        3. `Negative/` : non crack 이미지
        4. `split_info/` : train, test, validation 대상 이미지 경로가 맵핑된 txt 파일
    3. `SelectedCracks/`
        * 연구자 본인 기준에서 특징적이라고 판단한 이미지 65장
        * 선별기준 : 가로, 세로, 구석, 2줄로 나뉨,  회색 blob이 존재, crack이 비교적 두꺼움, 내부 분기를 만듦
    4. `test_images/` : 
        * test용 crack 이미지 3000장
        * 이미지 classification 결과는 `../test_results/`에 저장
    5. `UnseenData/`
        * coco + crack 으로 학습시킨 모델의 다중클래스 객체 탐지를 위한 데이터
        * 학습에 참여하지 않았지만, 인간이 판단 때 학습 클래스에 포함된다고 생각되는 데이터로 모델을 테스트해보기위한 데이터
2. `test_results/` : 하위 디렉터리에는 오분류된 이미지를 저장한다.
    1. `25/` : 학습된 모델에 confidence threshold=.25로 제한한 결과
    2. `75/` : 학습된 모델에 confidence threshold=.75로 제한한 결과
    3. `test crack and non crack.ipynb` : 각 threshold 별 분류 결과에 대한 시각화. confusion matrix 생성.
3. .ipynb 파일들
    1. `predict.ipynb` : jupyter 파일 내에서 터미널을 동작시켜 inference
    2. `revise yaml.ipynb`
        * python의 yaml 라이브러리를 활용해 .yaml파일을 수정
        * .yaml파일에 직접 수정할 수 없을 때 사용.
    3. `split dataset.ipynb` : 이미지를 목적에 맞게 .txt 파일로 분리
4. .py 파일들
    * crack 이미지 자동 라벨링을 위한 이미지 프로세싱 코드.
    * morphology 연산과 otsu 알고리즘을 사용
    1. `box.py` : cv2.rectangle을 이용해 bbox를 그림
    2. `contour.py` : cv2.findContours를 활용해 crack의 윤곽을 확인하고 bbox를 그림
    3. `lab.py` : 이미지 프로세싱 절차를 실험하기위한 실험실
    4. `test.py` : labelling 결과 bbox 좌표를 원본이미지에 재현하여, 자동 labelling의 정상작동 여부를 판단
        
## train, inference를 위한 간단한 터미널 명령어
: 아래 예시는 간단한 학습과 검출을 위한 명령어이며, 좀 더 풍부한 기능을 위해 사용자가 원하는 파라미터를 변경 및 설정할 수 있다.
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
