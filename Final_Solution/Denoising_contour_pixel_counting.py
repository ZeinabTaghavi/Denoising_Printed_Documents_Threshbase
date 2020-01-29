# Zeinab Taghavi
# avg of time = 3.108232495

import cv2
import numpy as np
from PIL import Image


def correct_rotation(img,max_word_height):# max_word_height = 140

    env_img = cv2.bitwise_not(img)
    env_img = cv2.dilate(env_img, kernel=np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(env_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # position of contour
        if h < max_word_height* 1.5:
            continue
        env_img[y:y + h, x:x + w] = 0
    
    img_temp = cv2.bitwise_not(env_img)
    edges = cv2.Canny(img_temp, 50, 150, apertureSize=3)
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


def move_images(img, img_source , borders , image_shape): # borders = [top, right, down, left] image_shape = [min_w, min_h]
    
    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    output_base_image = cv2.bitwise_not(img_source_gray)
    output_base_image = cv2.dilate(output_base_image ,kernel=np.ones((5,5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(output_base_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # the image or words must be between 10% and 90% of main image
    max_y = img.shape[0] - borders[2]
    max_x = img.shape[1] - borders[1]
    min_y = borders[0]
    min_x = borders[3]
    min_w = image_shape[0]
    min_h = image_shape[1]
    
    img_images = np.zeros((img.shape[0],img.shape[1]),np.uint8)

    # for any contour in main image , check if it is big enough to ba an image or word
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # position of contour

        if min_y < y and y+h < max_y  and min_x < x and x+w < max_x and w > min_w and h > min_h:
            print (x, y, w, h, min_y < y < max_y, min_x < x < max_x, w > min_w, h > min_h)
            img[y:y + h, x:x + w] = img_source_gray[y:y + h, x:x + w]
            img_images[y:y + h, x:x + w] = img_source_gray[y:y + h, x:x + w]
    print (img.shape , max_y ,max_x , min_y, min_x , min_w ,min_h  )
    cv2.imwrite(file_name +'-5-moved_just_images.jpg', img_images)
    return img



def remove_wasted_round_area(img, file_name, borders ,first_kernel_erod=(3,3), first_kernel_dilate=(15, 15),pixels_per_slice=20,
                             block_thresh=10, min_contour_area_minimizes_img=2,
                             last_kernel_dilate=(10, 10)): # borders = [top, right, down, left]
    print(img.shape)
    img[img.shape[0] - borders[2]:,:]=255
    img[:,img.shape[1] - borders[1]:]=255
    img[:borders[0],:]=255
    img[:,0:borders[3]] = 255

    cv2.imwrite(file_name + '-2-0-1-find_wasted_round_area_in_documents-borders.jpg', img)
    gray_env = cv2.bitwise_not(img)
    gray_env_erod = cv2.erode(gray_env, kernel=np.ones(first_kernel_erod, np.uint8), iterations=1)
    # cv2.imwrite(file_name + 'area_erode.jpg', gray_env_erod)
    gray_env_dilate = cv2.dilate(gray_env_erod, kernel=np.ones(first_kernel_dilate, np.uint8), iterations=1)
    # cv2.imwrite(file_name + 'area_dilate.jpg', gray_env_dilate)
    cv2.imwrite(file_name +'-2-0-find_wasted_round_area_in_documents-gray_env_dilate_.jpg', gray_env_dilate)

    poly = np.zeros((int(gray_env_dilate.shape[0] / pixels_per_slice), int(gray_env_dilate.shape[1] / pixels_per_slice), 1), np.uint8)
    poly.fill(0)
    pices = (int(gray_env_dilate.shape[0] / pixels_per_slice), int(gray_env_dilate.shape[1] / pixels_per_slice))
    for y in range(pices[0]):
        for x in range(pices[1]):
            poly[y, x] = np.mean(gray_env_dilate[(y * pixels_per_slice):((y + 1) * pixels_per_slice), (x * pixels_per_slice):((x + 1) * pixels_per_slice)])
    cv2.imwrite('poly_temp_not_threshed.jpg',poly)
    _, poly = cv2.threshold(poly, block_thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite('poly_temp_after_threshed.jpg', poly)

    # cv2.imwrite(file_name +'-2-1-find_wasted_round_area_in_documents-poly_.jpg', poly)

    # cv2.imwrite(file_name +'-2-2-find_wasted_round_area_in_documents-poly_.jpg', poly)


    contours, _ = cv2.findContours(poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area_minimizes_img:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(poly, (x, y), (x + w, y + h), 0, -1)

    poly = cv2.dilate(poly, kernel=np.ones(last_kernel_dilate, np.uint8), iterations=1)
    cv2.imwrite(file_name +'-2-3-find_wasted_round_area_in_documents-poly-contoured_.jpg', poly)


    poly3 = np.zeros((int(gray_env_dilate.shape[0]), int(gray_env_dilate.shape[1]), 1), np.uint8)
    poly3.fill(0)
    for y in range(0, pices[0]):
        for x in range(0, pices[1]):
            poly3[(y * pixels_per_slice):((y + 1) * pixels_per_slice), (x * pixels_per_slice):((x + 1) * pixels_per_slice)] = poly[y, x]

    # cv2.imwrite(file_name +'-2-4-find_wasted_round_area_in_documents-poly3_.jpg', poly3)

    no_waisted_area = cv2.bitwise_not(poly3)
    no_waisted_area_on_source = cv2.bitwise_or(img, no_waisted_area)

    # cv2.imwrite(file_name +'-2-5-find_wasted_round_area_in_documents_.jpg', no_waisted_area)

    return no_waisted_area_on_source,no_waisted_area



def denoise_by_contours(img, img_source, file_name,blur_size=(3, 3), denoise_erode_kernel=(2, 2),
                        borders=[100,100,100,100], contour_min_area=30, min_noise_h=150, max_noise_w = 30):

    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray_img = img


    if blur_size[0]!= 0 and blur_size[1]!=0:
        blur = cv2.blur(gray_img, blur_size)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = gray_img

    gray_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=np.ones(denoise_erode_kernel, np.uint8))
    # cv2.imwrite( file_name +'-1-1-first_sort_of_noises.jpg', gray_img)

    contour, _ = cv2.findContours(gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    new_img.fill(255)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = 1.23456789e-14
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if w > img.shape[1]-borders[1]-borders[3] or h > img.shape[0]-borders[1] - borders[3]:
            continue

        if cv2.contourArea(cnt) < contour_min_area:
            continue

        if h > w and h > min_noise_h and w < max_noise_w:
            continue

        cv2.fillPoly(new_img, [approx], (0, 0, 0))

    # cv2.imwrite(file_name + '-1-2-newimage_as_temp.jpg', new_img)
    mask = cv2.bitwise_or(new_img, gray_img)

    # background is inverse => dilate -> erode
    # cv2.imwrite(file_name + '-1-3-mask.jpg', mask)

    mask_dilate = cv2.erode(mask, kernel=np.ones(blur_size, np.uint8))
    # cv2.imwrite(file_name +'-1-4-mask_dilate.jpg', mask_dilate)

    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_or(mask_dilate, img_source_gray)

    return img


if __name__ =='__main__':
    time = []
    max_word_height = 190  # max height of line
    borders = [0, 250, 350, 200] # borders = [top, right, down, left]
    image_dir = './'
    for i in range(1,13):
        e1 = cv2.getTickCount()
        file_name = image_dir+'/'+'image'+str(i)+'.tif'
        img_source = cv2.imread(file_name)

        # for first denoising by contour

        blur_size = (0, 0)  # less than dot's diameter
        denoise_erode_kernel = (0, 0)  # same as blur
        contour_min_area = 12  # less than dot's are
        min_noise_h = 150  # more than biggest lines height
        max_noise_w = 3  # less than thinnest acceptable line

        denoised_by_contours = denoise_by_contours(img_source,img_source,file_name,
                                                   blur_size=blur_size,
                                                   denoise_erode_kernel=denoise_erode_kernel,
                                                   borders=borders,
                                                   contour_min_area=contour_min_area,
                                                   min_noise_h=min_noise_h,
                                                   max_noise_w=max_noise_w)

        cv2.imwrite(file_name + '-1-denoise_by_contours.jpg', denoised_by_contours)

        first_kernel_erod = (3, 3) # less than dot's diameter
        max_space = 30
        min_word_height = 30
        first_kernel_dilate = (int(min_word_height/4), max_space)
        pixels_per_slice = 15
        block_thresh = 20
        min_contour_area_minimizes_img = int((2*(max_space+min_word_height)) / pixels_per_slice)
        last_kernel_dilate = (10, 10)

        removed_wasted_round_area, pattern = remove_wasted_round_area(denoised_by_contours, file_name, borders,
                                                                      first_kernel_erod=first_kernel_erod,
                                                                      first_kernel_dilate=first_kernel_dilate,
                                                                      pixels_per_slice=pixels_per_slice,
                                                                      block_thresh=pixels_per_slice,
                                                                      min_contour_area_minimizes_img=min_contour_area_minimizes_img,
                                                                      last_kernel_dilate=last_kernel_dilate
                                                                      )

        cv2.imwrite(file_name + '-2-removed_wasted_round_area_pattern.jpg', pattern)
        cv2.imwrite(file_name + '-2-removed_wasted_round_area_img.jpg', removed_wasted_round_area)

        image_shape = [500,500]
        returned_images_img = move_images(removed_wasted_round_area , img_source, borders, image_shape)
        cv2.imwrite(file_name + '-5-returned_images.jpg', returned_images_img)

        corrected_rotation = correct_rotation(returned_images_img, max_word_height)
        cv2.imwrite(file_name + '-6-corrected_rotation.jpg', corrected_rotation)
        e2 = cv2.getTickCount()
        time.append((e2-e1)/cv2.getTickFrequency())

    print(np.mean(time))
