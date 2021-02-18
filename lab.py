import os

import cv2
import numpy as np
'''
image processing algorithms : https://d2.naver.com/helloworld/8344782
opencv lecture-kor : https://076923.github.io/posts/#Python-OpenCV
opencv documentation : https://docs.opencv.org/master/
'''

if __name__ == "__main__":
    dir = os.path.join(os.getcwd() + '\data\Positive')
    file_name = '00019.jpg'
    file_path = os.path.join(dir, file_name)  #  19042_1, 00926 5617 6221 4892
    original_image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # _, vanilla_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # _, otsu_binary = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)
    # _, triangle_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRIANGLE)

    # remove noise
    # 1. dilaton + erosion 3 : 노이즈 제거 후 crack 부각
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dilation = cv2.dilate(otsu_binary, kernel, anchor=(-1, -1), iterations=1)  # 하얀 부분이 팽창 -> crack 수축 : 얇은 나뭇잎 제거에 효과적
    # erosion = cv2.erode(dilation, kernel, anchor=(-1, -1), iterations=2)  # 하얀 부분이 수축. dilate 후 erode로 줄어든 크기 복구
    # cv2.imshow('erosion 1', erosion)

    # 2. erode(otsu) + 크기로 노이즈 제거 : crack 부각 후 노이즈 제거
    # erosion = cv2.erode(otsu_binary, kernel, anchor=(-1, -1), iterations=1)  # 얇은 실 선으로 이어진 crack들을 하나로 검출하는데 효과적
    # cv2.imshow('erosion 2', erosion)

    img_ero = gray_image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_ero = cv2.erode(img_ero, kernel, iterations=10)

    img_dil = img_ero.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    img_dil = cv2.dilate(img_dil, kernel, iterations=10)

    t, t_otsu = cv2.threshold(img_dil, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # binary = cv2.bitwise_not(erosion)

    contours, hierarchy = cv2.findContours(t_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # VERTICAL BOX
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 20 or h <= 20:  # 너무 작은 crack 혹은 noise 제거
            continue
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(original_image, (x, y), 3, (255, 0, 0), cv2.FILLED)
        cv2.circle(original_image, (x+int(w/2), y), 3, (255, 0, 0), cv2.FILLED)
        cv2.circle(original_image, (x, y+int(h/2)), 3, (255, 0, 0), cv2.FILLED)

    # ROTATED BOX
    # for cnt in contours:
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(original_image, [box], 0, (0, 0, 255), 2)

    # ONLY CONTOUR
    # for i in range(len(contours)):
    #     cv2.drawContours(original_image, [contours[i]], 0, (0, 0, 255), 2)

    # SHOW CONOUTR RESULT
    print(file_path)
    cv2.imshow(os.path.basename(file_path).split('.')[0], original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
