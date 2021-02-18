import os, copy
import contour

import cv2

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)
    result_dir = os.path.join(BASE_DIR, 'result')
    original_data_dir = os.path.join(BASE_DIR, 'data/Positive')
    test_dir1 = os.path.join(BASE_DIR, 'test result1')
    test_dir2 = os.path.join(BASE_DIR, 'test result2')

    if not os.path.isdir(test_dir1):
        os.mkdir(test_dir1)

    if not os.path.isdir(test_dir2):
        os.mkdir(test_dir2)

    file_list = os.listdir(result_dir)

    for text_file in file_list:
        file_name = text_file.split('.')[0]
        test_image = cv2.imread(os.path.join(original_data_dir, file_name+'.jpg'), flags=cv2.IMREAD_COLOR)

        # 텍스트 파일로 bounding box 역추적
        with open(os.path.join(result_dir, text_file), mode='r') as f:
            lines = f.readlines()
            for line in lines:
                center_x, center_y, width, height = map(int, line.split())
                lefttop = (center_x - int(width/2), center_y - int(height/2))
                rightbottom = (center_x + int(width/2), center_y + int(height/2))
                cv2.rectangle(test_image, lefttop, rightbottom, (255, 0, 0), 5, cv2.LINE_8)

        # 원래 알고리즘으로 그리기
        original_image = cv2.imread(os.path.join(original_data_dir, file_name + '.jpg'), flags=cv2.IMREAD_COLOR)
        binaries = contour.get_binaries(original_image)
        original_image, _ = contour.draw_contour(file_name, binaries, original_image, setting=1)  # 1 : vertical box, 2: rotated box, 3 : contour

        # 결과 저장
        cv2.imwrite(os.path.join(test_dir1, file_name+'.jpg'), test_image)
        cv2.imwrite(os.path.join(test_dir2, file_name+'.jpg'), original_image)