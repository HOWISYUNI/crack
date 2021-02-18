import os, copy

import cv2
import numpy as np
'''
image processing algorithms : https://d2.naver.com/helloworld/8344782
opencv lecture-kor : https://076923.github.io/posts/#Python-OpenCV
opencv documentation : https://docs.opencv.org/master/
'''


def write_points(file_path, points):
    SAVING_DIR = os.path.join(os.getcwd(), 'points')
    file_name = os.path.basename(file_path).split('.')[0]
    IMAGE_SIZE = 227

    if not os.path.isdir(SAVING_DIR):
        os.mkdir(SAVING_DIR)

    for x, y, w, h in points:

        center_x = (x + w/2) / IMAGE_SIZE
        center_y = (y + h/2) / IMAGE_SIZE
        w = w / IMAGE_SIZE
        h = h / IMAGE_SIZE

        with open(os.path.join(SAVING_DIR, file_name+'.txt'), mode='a') as f:
            f.write(f'0 {center_x} {center_y} {w} {h}\n')


def save_image(file_path, saving_dir, original_image):
    SAVING_DIR = saving_dir
    file_name = os.path.basename(file_path)

    if not os.path.isdir(SAVING_DIR):
        os.mkdir(SAVING_DIR)

    cv2.imwrite(os.path.join(SAVING_DIR, file_name), original_image)


def is_overlay(overlay_history, x, y, w, h):
    comparison_target = np.zeros(overlay_history.shape)
    comparison_target[x:x+w, y:y+h] = 1
    overlay_result = overlay_history * comparison_target

    return bool(np.where(overlay_result > 0)[0].size)


def draw_contour(file_path, binaries, original_image, setting=1):
    """
    요구사항
    1. box 중심 x, y
    2. box width, height
    """

    box_counts = [0, 0]
    IMAGE_SIZE = 227
    pure_image = copy.deepcopy(original_image)

    for idx, binary in enumerate(binaries):
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if setting == 1:  # VERTICAL RECTANGLE
            saving_dir = os.path.join(os.path.dirname(__file__), 'image processing results/final')
            overlay_history = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
            points = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                if w <= 60 and h <= 60:  # 너무 작은 crack 혹은 noise 제거
                    continue

                if cv2.arcLength(cnt, True) < 300:  # contour 길이가 300 이하면 = 작으면 제외
                    continue

                if is_overlay(overlay_history, x, y, w, h):  # 겹치는 경우, 박스없는 순수한 이미지를 ./results/final/overlay 폴더에 저장
                    print('=========================== overlay')
                    saving_dir = os.path.join(os.path.dirname(__file__), 'image processing results/final/overlay')
                    save_image(file_path, saving_dir, pure_image)
                    break

                overlay_history[x:x+w, y:y+h] = 1
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box_counts[idx] += 1
                points.append((x, y, w, h))

            else:
                if box_counts[0] == 0:  # bounding box가 하나도 그려지지 않은 경우
                    print('=========================== None')
                    saving_dir = os.path.join(os.path.dirname(__file__), 'image processing results/final/None')
                    save_image(file_path, saving_dir, original_image)

                else:
                    save_image(file_path, saving_dir, original_image)
                    write_points(file_path, points)

        elif setting == 2:  # ROTATED RECTANGLE
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(original_image, [box], 0, (0, 0, 255), 2)

        elif setting == 3:  # ONLY CONTOURS
            for i in range(len(contours)):
                cv2.drawContours(original_image, [contours[i]], 0, (0, 0, 255), 2)
                # print(i, hierarchy[0][i])

    return original_image, box_counts


def get_binaries(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # _, vanilla_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # _, otsu_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)

    # REMOVE NOISE
    erosions = []
    # 1. dilaton + erosion 3 : 노이즈 제거 후 crack 부각
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dilation = cv2.dilate(otsu_binary, kernel, anchor=(-1, -1), iterations=1)  # 하얀 부분이 팽창 -> crack 수축 : 얇은 나뭇잎 제거
    # type1 = cv2.erode(dilation, kernel, anchor=(-1, -1), iterations=3)  # 하얀 부분이 수축. dilate 후 erode로 줄어든 크기 복구
    # erosions.append(type1)

    # 2. erode(otsu) + 크기로 노이즈 제거 : crack 부각 후 노이즈 제거
    # type2 = cv2.erode(otsu_binary, kernel, anchor=(-1, -1), iterations=1)  # 얇은 실 선으로 이어진 crack들을 하나로 검출하는데 효과적
    # erosions.append(type2)

    # ---------------------------------------혜원------------------------------------------------------------------------------
    erosion_image = gray_image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    erosion_image = cv2.erode(erosion_image, kernel, iterations=10)

    dilation_image = erosion_image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilation_image = cv2.dilate(dilation_image, kernel, iterations=20)

    _, otsu_binary = cv2.threshold(dilation_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    erosions.append(otsu_binary)
    # -------------------------------------------------------------------------------------------------------------------------
    # temp = gray_image.copy()
    # ===================== 후보 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # opening = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
    # _, otsu_binary = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ===================== 후보 2
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # opening = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
    # temp2 = opening.copy()
    # closing = cv2.morphologyEx(temp2, cv2.MORPH_CLOSE, kernel)
    # _, otsu_binary = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # erosions.append(otsu_binary)

    # CONVERT BLACK <-> WHITE
    binaries = []
    for erosion in erosions:
        binaries.append(erosion)

    return binaries


def make_box(file_path, contour_type=1):
    # READ IMAGE
    print(file_path)
    original_image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)

    # GET BINARY IMAGE
    binaries = get_binaries(original_image)

    # DRAW and get BOX COUNT RESULTS
    original_image, box_counts = draw_contour(file_path, binaries, original_image, setting=contour_type)  # 1 : vertical box, 2: rotated box, 3 : contour

    # SHOW CONTOUR RESULT
    # cv2.imshow(os.path.basename(file_path), original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array(box_counts)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(__file__)
    target_dir = os.path.join(BASE_DIR, 'Original Data/Selected/Positive')

    results = np.array([0, 0])

    for _, dirs, files in os.walk(target_dir):
        if dirs:
            for file in files:
                count_reuslt = make_box(os.path.join(target_dir, file), contour_type=1)
                results = np.vstack((results, count_reuslt))

    # results = make_box(r'C:\Users\piai\Desktop\pycharm_projects\Crack\data\Positive\00102.jpg', contour_type=1)
    # results = results.reshape(-1, 2)

    print(results.mean(axis=0))
