import cv2
import numpy as np

if __name__ == "__main__":
    # file_path = './data/Positive/05704.jpg' # 나뭇잎 noise 156 5704
    file_path = './data/Positive/05381.jpg' # crack 이 조각난 경우 5617 6221
    # file_path = './data/Positive/07060.jpg'  # crack 이 조각난 경우
    original_image = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
    # gray_image = cv2.imread(file_path, flags=cv2.IMREAD_GRAYSCALE) # 1채널, 그레이스케일 적용
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    _, vanila_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    _, otsu_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)
    _, triangle_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRIANGLE)
    thr1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilate = cv2.dilate(otsu_binary, kernel, anchor=(-1, -1), iterations=2)

    y, x = np.where(dilate == 0)
    left, right, top, bottom = np.min(x), np.max(x), np.min(y), np.max(y)
    lefttop = (left, top)
    rightbottom = (right, bottom)

    # lefttop = tuple(np.flip(result[0]))
    # rightbottom = tuple(np.flip(result[-1]))

    # cv2.rectangle(otsu_binary, lefttop, rightbottom, (0,255,0), 3, cv2.LINE_AA)
    cv2.rectangle(original_image, lefttop, rightbottom, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('original', original_image)
    # cv2.imshow('threshold', vanila_binary)
    # cv2.imshow('otsu threshold', otsu_binary)
    # cv2.imshow('triangle threshold', triangle_binary)
    # cv2.imshow('adaptive_mean', thr1)
    # cv2.imshow('adaptive_gaussian', thr2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()