import cv2
import numpy as np
import glob


def correct_rotation(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    num = 0
    sum = 0

    ## if there is no line, or OCR could not find
    try:
        for i in lines:
            for rho, theta in i:
                if np.degrees(theta) > 45 and np.degrees(theta) < 135:
                    sum += np.degrees(theta)
                    num += 1

        rows, cols = img.shape[0], img.shape[1]
        if num != 0:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (sum / num) - 90, 1)
            img = cv2.warpAffine(img, M, (cols, rows))
            theta_radian = np.radians((sum / num) - 90)
            y = int(np.sin(theta_radian) / np.cos(theta_radian) * img.shape[0])
            img[0:abs(y), :] = 255
            img[img.shape[0] - abs(y):, :] = 255
            x = int(np.sin(theta_radian) / np.cos(theta_radian) * img.shape[1])
            img[:, 0:abs(y)] = 255
            img[:, img.shape[1] - abs(y):] = 255
    except:
        pass
    y_border_size = int(img.shape[0]*(.05))
    x_border_size = int(img.shape[1]*(.05))
    img[0:y_border_size, :] = 255
    img[img.shape[0] - y_border_size: img.shape[0], :] = 255
    img[:, 0:x_border_size] = 255
    img[:, img.shape[1] - x_border_size:img.shape[1]] = 255
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img, mask, (0, 0), (255,255,255))

    return img

if __name__ == "__main__":
    for file_name in glob.glob('*.bmp'):
        img = cv2.imread(file_name)
        img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        corrected_rotation = correct_rotation(img_gray)
        cv2.imwrite('final'+file_name+'_.jpg',  corrected_rotation)

