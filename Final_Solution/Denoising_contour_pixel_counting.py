# Zeinab Taghavi
# avg of time = 3.108232495

import cv2
import numpy as np
from PIL import Image


# correct rotation will find the main lines in pages and then rotate image to the average of thetas
def correct_rotation(img,max_word_height):# max_word_height = 140
    
    env_img = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(env_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # position of contour
        if h < max_word_height* 1.5:
            continue
        env_img[y:y + h, x:x + w] = 0
    
    img = cv2.bitwise_not(env_img)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    num = 0
    sum = 0

    # if there is no line
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
    
    return img


# move_images will replace image in denoised_image in order to save image from changind
def move_images(img, img_source , borders): # borders = [top, right, down, left]
    
    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    output_base_image = cv2.bitwise_not(img_source_gray)
    contours, _ = cv2.findContours(output_base_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # the image or words must be between 10% and 90% of main image
    max_height = img.shape[0] - borders[2]
    max_width = img.shape[1] - borders[1]
    min_height = borders[0]
    min_width = borders[3]

    # for any contour in main image , check if it is big enough to ba an image or word
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # position of contour
        if w > max_width or w < min_width or h > max_height or h < min_height or y + h > max_height or x + w > max_width or y<min_height or x < min_width:
            continue
        img[y:y + h, x:x + w] = img_source_gray[y:y + h, x:x + w]

    cv2.imwrite(file_name +'-5-move_images.jpg', img)
    return img


def remove_wasted_round_area(img, file_name, borders ,first_kernel_erod=(3,3), first_kernel_dilate=(15, 15),pixels_per_slice=20,
                             making_square_thresh=10, border_pixel=5, min_contour_area_minimizes_img=2,
                             last_kernel_dilate=(10, 10)): # borders = [top, right, down, left]
    # 0 - removing_borders
    img[img.shape[0] - borders[2]:,:]=255
    img[:,img.shape[1] - borders[1]:]=255
    img[:borders[0],:]=255
    img[:,0:borders[3]] = 255
    
    # 1 - we need dpi for slicing image
    imgPIL = Image.open(file_name)
    dpi = (300, 300)  # default is (300 , 300)
    if 'dpi' in imgPIL.info.keys():
        if imgPIL.info['dpi'] != 0:
            dpi = imgPIL.info['dpi']
    del imgPIL

    # 2 - use erod nad then dilate in order to clear small noises
    gray_env = cv2.bitwise_not(img)
    gray_env_erod = cv2.erode(gray_env, kernel=np.ones(first_kernel_erod, np.uint8), iterations=1)
    gray_env_dilate = cv2.dilate(gray_env_erod, kernel=np.ones(first_kernel_dilate, np.uint8), iterations=1)

    # 3 - by pixel_counting way we want to find wasted areas
    slice = int(dpi[0] / pixels_per_slice)

    cv2.imwrite(file_name +'-2-0-find_wasted_round_area_in_documents-gray_env_dilate_.jpg', img)

    poly = np.zeros((int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice), 1), np.uint8)
    poly.fill(0)
    pices = (int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice))
    for y in range(pices[0]):
        for x in range(pices[1]):
            poly[y, x] = np.mean(gray_env_dilate[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)])
    _, poly = cv2.threshold(poly, making_square_thresh, 255, cv2.THRESH_BINARY)


    # cv2.imwrite(file_name +'-2-1-find_wasted_round_area_in_documents-poly_.jpg', poly)

    poly[0:border_pixel, :] = 255
    poly[poly.shape[0] - border_pixel: poly.shape[0], :] = 255
    poly[:, 0:border_pixel] = 255
    poly[:, poly.shape[1] - border_pixel:poly.shape[1]] = 255
    h, w = poly.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(poly, mask, (0, 0), 0)

    # cv2.imwrite(file_name +'-2-2-find_wasted_round_area_in_documents-poly_.jpg', poly)

    contours, _ = cv2.findContours(poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area_minimizes_img:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(poly, (x, y), (x + w, y + h), 0, -1)

    # cv2.imwrite(file_name +'-2-3-find_wasted_round_area_in_documents-poly-contoured_.jpg', poly)

    poly = cv2.dilate(poly, kernel=np.ones(last_kernel_dilate, np.uint8), iterations=1)
    poly3 = np.zeros((int(gray_env_dilate.shape[0]), int(gray_env_dilate.shape[1]), 1), np.uint8)
    poly3.fill(0)
    for y in range(0, pices[0]):
        for x in range(0, pices[1]):
            poly3[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)] = poly[y, x]

    # cv2.imwrite(file_name +'-2-4-find_wasted_round_area_in_documents-poly3_.jpg', poly3)

    no_waisted_area = cv2.bitwise_not(poly3)
    no_waisted_area_on_source = cv2.bitwise_or(img, no_waisted_area)

    # cv2.imwrite(file_name +'-2-5-find_wasted_round_area_in_documents_.jpg', no_waisted_area)

    return no_waisted_area_on_source,no_waisted_area


def denoise_by_contours(img, img_source, blur_size=(3, 3), denoise_erode_kernel=(2, 2),
                        max_contour_percent=[.05, .05], contour_min_area=30, min_noise_h=150, max_noise_w = 30,
                        last_dilate_kernel=(2,2), last_dilate_iteration=4 ,
                        last_erode_kernel=(0,0), last_erode_iteration=0):

    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray_img = img

    blur = cv2.blur(gray_img, blur_size)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    gray_img = cv2.dilate(thresh, kernel=np.ones(denoise_erode_kernel, np.uint8))
    contour, _ = cv2.findContours(gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    new_img.fill(255)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = 1.23456789e-14
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if w > img.shape[1] * ( 1 - max_contour_percent[1]) or h > img.shape[0] * (1 - max_contour_percent[0]):
            continue

        if cv2.contourArea(cnt) < contour_min_area:
            continue

        if h > w and h > min_noise_h and w < max_noise_w:
            continue

        cv2.fillPoly(new_img, [approx], (0, 0, 0))
    gray_img = cv2.bitwise_or(new_img, gray_img)

    # background is inverse => dilate -> erode
    img_env_erode = cv2.erode(gray_img, kernel=np.ones((last_dilate_kernel), np.uint8), iterations=last_dilate_iteration)

    img_env_dilate = cv2.dilate(img_env_erode, kernel=np.ones(last_erode_kernel, np.uint8), iterations=last_erode_iteration)

    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_or(img_env_dilate, img_source_gray)

    return img


if __name__ =='__main__':
    time = []
    max_word_height = 140
    borders = [489, 560, 477, 630] # borders = [top, right, down, left]
    
    for i in range(1,2):
        e1 = cv2.getTickCount()
        file_name = 'image'+str(i)+'.bmp'
        img_source = cv2.imread(file_name)

        # for first denoising by contour
        blur_size = (3, 3)
        denoise_erode_kernel = (4, 4)
        max_contour_percent = [.05, .05]
        contour_min_area = 30
        min_noise_h = 150
        max_noise_w = 50
        last_dilate_kernel = (4, 4)
        last_dilate_iteration = 4
        last_erode_kernel = (0, 0)
        last_erode_iteration = 0

        denoised_by_contours = denoise_by_contours(img_source,img_source,
                                                   blur_size=blur_size,
                                                   denoise_erode_kernel=denoise_erode_kernel,
                                                   max_contour_percent=max_contour_percent,
                                                   contour_min_area=contour_min_area,
                                                   min_noise_h=min_noise_h,
                                                   max_noise_w=max_noise_w,
                                                   last_dilate_kernel=last_dilate_kernel,
                                                   last_dilate_iteration=last_dilate_iteration,
                                                   last_erode_kernel=last_erode_kernel,
                                                   last_erode_iteration=last_erode_iteration)

        cv2.imwrite(file_name + '-1-denoise_by_contours.jpg', denoised_by_contours)

        # # for removing waster round area
        first_kernel_erod = (5, 5)
        first_kernel_dilate = (15, 15)
        pixels_per_slice = 20
        making_square_thresh = 20
        border_pixel = 5
        min_contour_area_minimizes_img = 1
        last_kernel_dilate = (10, 10)

        removed_wasted_round_area , pattern = remove_wasted_round_area(denoised_by_contours,file_name,borders,
                                                                       first_kernel_erod=first_kernel_erod,
                                                                       first_kernel_dilate=first_kernel_dilate,
                                                                       pixels_per_slice=pixels_per_slice,
                                                                       making_square_thresh=pixels_per_slice,
                                                                       border_pixel=border_pixel,
                                                                       min_contour_area_minimizes_img=min_contour_area_minimizes_img,
                                                                       last_kernel_dilate=last_kernel_dilate
                                                                       )

        cv2.imwrite(file_name + '-2-removed_wasted_round_area_pattern.jpg', pattern)
        cv2.imwrite(file_name + '-2-removed_wasted_round_area_img.jpg', removed_wasted_round_area)

        # for second denoising by contour '''''if needed '''''
        blur_size = (1, 1)
        denoise_erode_kernel = (0, 0)
        max_contour_percent = [.0, .0]
        contour_min_area = 70
        min_noise_h = 0
        max_noise_w = 0
        last_dilate_kernel = (6, 6)
        last_dilate_iteration = 1
        last_erode_kernel = (2, 2)
        last_erode_iteration = 1

        denoised_by_contours_again = denoise_by_contours(removed_wasted_round_area,img_source,
                                                         blur_size=blur_size,
                                                         denoise_erode_kernel=denoise_erode_kernel,
                                                         max_contour_percent=max_contour_percent,
                                                         contour_min_area=contour_min_area,
                                                         min_noise_h=min_noise_h,
                                                         max_noise_w=max_noise_w,
                                                         last_dilate_kernel=last_dilate_kernel,
                                                         last_dilate_iteration=last_dilate_iteration,
                                                         last_erode_kernel=last_erode_kernel,
                                                         last_erode_iteration=last_erode_iteration
                                                         )

        cv2.imwrite(file_name + '-3-denoise_by_contours_again.jpg', denoised_by_contours_again)

        # # for removing waster round area again
        first_kernel_erod = (5, 5)
        first_kernel_dilate = (30, 30)
        pixels_per_slice = 20
        making_square_thresh = 10
        border_pixel = 0
        min_contour_area_minimizes_img = 7
        last_kernel_dilate = (5, 5)

        removed_wasted_round_area_again, pattern = remove_wasted_round_area(denoised_by_contours_again,file_name,borders,
                                                                            first_kernel_erod=first_kernel_erod,
                                                                            first_kernel_dilate=first_kernel_dilate,
                                                                            pixels_per_slice=pixels_per_slice,
                                                                            making_square_thresh=pixels_per_slice,
                                                                            border_pixel=border_pixel,
                                                                            min_contour_area_minimizes_img=min_contour_area_minimizes_img,
                                                                            last_kernel_dilate=last_kernel_dilate
                                                                            )

        cv2.imwrite(file_name + '-4-removed_wasted_round_area_again_pattern.jpg', pattern)
        cv2.imwrite(file_name + '-4-removed_wasted_round_area_again_img.jpg', removed_wasted_round_area_again)

        returned_images_img = move_images(removed_wasted_round_area_again , img_source, borders)
        cv2.imwrite(file_name + '-5-returned_images.jpg', returned_images_img)

        corrected_rotation = correct_rotation(returned_images_img, max_word_height)
        cv2.imwrite(file_name + '-6-corrected_rotation.jpg', corrected_rotation)
        e2 = cv2.getTickCount()
        time.append((e2-e1)/cv2.getTickFrequency())

    print(np.mean(time))
