# zeinab Taghavi
#
# time: between 6.9s and 9.3s
#
# 1 - erod and dilate for removing small noises
# 2 - correct rotation to more accurately find lines
# 3 - find the high compression vertical area
# 4 - in vertical high compression areas, make all horizontal high compression areas
# 5 - have a light dilate in white background to remove small noises
# 6 - using hOCR for denoised image


import cv2
import numpy as np
from lxml import etree
import pytesseract

def move_mser_to_new_image(img): # gray image

    destination = np.zeros((img.shape[0],img.shape[1]) , np.uint8)
    destination.fill(255)
    mser =cv2.MSER_create()
    regions = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]  # convert them to list that can be used for polygons
    filled_image = cv2.fillPoly(destination, hulls, 0, 0)  # line polygons around words
    final = cv2.bitwise_or(img , filled_image)
    return final

def find_line_by_semi_histogram(img_file):

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
        y_border_size = int(img.shape[0] * (.05))
        x_border_size = int(img.shape[1] * (.05))
        img[0:y_border_size, :] = 255
        img[img.shape[0] - y_border_size: img.shape[0], :] = 255
        img[:, 0:x_border_size] = 255
        img[:, img.shape[1] - x_border_size:img.shape[1]] = 255
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(img, mask, (0, 0), (255, 255, 255))

        return img


    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1 - erod and dilate for removing small noises

    gray_env = cv2.bitwise_not(gray)
    kernel_erod = np.ones((5, 7), np.uint8)
    gray_env_erod = cv2.erode(gray_env, kernel_erod, iterations=1)

    kernel_dilate = np.ones((15, 20), np.uint8)
    img = cv2.dilate(gray_env_erod, kernel_dilate, iterations=2)
    img = cv2.bitwise_not(img)

    # 2 - correct rotation to more accurately find lines

    corrected_rotation =  correct_rotation(img)
    gray_corrected_rotation = corrected_rotation.copy()
    gray_env = cv2.bitwise_not(gray_corrected_rotation)
    # cv2.imwrite('Denoising_semiHistogram_1_eros_dilated_and_corrected_rotation.jpg' , gray_env)

    # 3 - find the high compression vertical area

    vertical_hist = [sum(gray_env[i,:]) for i in range(corrected_rotation.shape[0])]
    vertical_temp = gray_corrected_rotation.copy()
    vertical_limit = gray_env.shape[1] * 255 * .01
    for i in range(len(vertical_hist)):
        if vertical_hist[i] > vertical_limit:
            vertical_temp[i,:] = 255
        else:
            vertical_temp[i,:] = 0

    # cv2.imwrite('Denoising_semiHistogram_hOCR_2_vertical_line_detected.jpg' , vertical_temp)
    contour , _ = cv2.findContours(vertical_temp , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(corrected_rotation , contour , -1 , 100 , 3)

    # 4 - in vertical high compression areas, make all horizontal high compression areas

    vertical_lines_positions = []  # they are vertical high compression areas
    for cnt in contour:
        x , y , w , h = cv2.boundingRect(cnt)
        vertical_lines_positions.append([y,y+h])
        # corrected_rotation = cv2.rectangle(corrected_rotation , (x,y) , (x+w , y+h) , (0,0,200) ,-1)
    # cv2.imwrite('Denoising_semiHistogram_3_vertical_line_rected.jpg',corrected_rotation)

    gray_corrected_rotation_env = cv2.bitwise_not(gray_corrected_rotation)
    for y1,y2 in vertical_lines_positions:
        temp_img_env = gray_corrected_rotation_env[y1:y2,:]
        horizontal_limit = (y2-y1) * 255 * .1
        for j in range(temp_img_env.shape[1]):
            if sum(temp_img_env[:,j]) > horizontal_limit:
                temp_img_env[:, j] = 255
            else:
                temp_img_env[:, j] = 0
        temp_img = cv2.bitwise_not(temp_img_env)
        gray_corrected_rotation[y1:y2,:] = temp_img

    denoised = cv2.bitwise_or(gray_corrected_rotation ,gray )
    # cv2.imwrite('Denoising_semiHistogram_4_horizontal_line_rected.jpg', denoised)

    # 5 - have a light dilate in white background to remove small noises

    kernel_erod = np.ones((3, 3), np.uint8)
    denoised_dilated = cv2.dilate(denoised, kernel_erod, iterations=1)
    denoised_eroded = cv2.erode(denoised_dilated, kernel_erod, iterations=1)
    final = cv2.bitwise_or(gray_corrected_rotation , denoised_eroded)
    # cv2.imwrite('Denoising_semiHistogram_5_denoised.jpg', final)

    # cv2.imwrite( img_file +'Denoising_semiHistogram_MSER_5_denoised.jpg', final)

    moved_final = move_mser_to_new_image(final)
    cv2.imwrite(img_file+'Denoising_semiHistogram_MSER_6_MSER_moved.jpg' , moved_final)


    return final


if __name__ == '__main__':
    n1 = 1
    n2 = 2
    for i in range(n1,n2):
        e1 = cv2.getTickCount()
        img_file = 'image'+str(i)+'.bmp'
        find_line_by_semi_histogram(img_file)
        e2 = cv2.getTickCount()
        print(str(i)+' the time is: '+str((e2-e1)/cv2.getTickFrequency()))
