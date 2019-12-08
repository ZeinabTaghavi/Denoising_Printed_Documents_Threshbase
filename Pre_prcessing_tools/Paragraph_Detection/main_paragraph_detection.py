# avg of time =  2.7401173988

from lxml import etree
import cv2
from scipy import ndimage
import pytesseract
import numpy as np
from PIL import Image


def making_data_ready(n):
    img1 = cv2.imread(n + '.jpg')

    img2 = 255 - img1[650:650 + 200, 750:750 + 1800]
    rotated = ndimage.rotate(img2, 90)
    temp = img1.copy()
    temp[0:rotated.shape[0], 0:rotated.shape[1]] = cv2.bitwise_xor(img1[0:rotated.shape[0], 0:rotated.shape[1]],
                                                                   rotated)
    cv2.imwrite(n + '1.jpg', temp)

    rotated = ndimage.rotate(img2, 60)
    temp = img1.copy()
    temp[0:rotated.shape[0], 0:rotated.shape[1]] = cv2.bitwise_xor(img1[0:rotated.shape[0], 0:rotated.shape[1]],
                                                                   rotated)
    cv2.imwrite(n + '2.jpg', temp)

    rotated = ndimage.rotate(img2, 30)
    temp = img1.copy()
    temp[0:rotated.shape[0], 0:rotated.shape[1]] = cv2.bitwise_xor(img1[0:rotated.shape[0], 0:rotated.shape[1]],
                                                                   rotated)
    cv2.imwrite(n + '3.jpg', temp)

    rotated = ndimage.rotate(img2, 15)
    temp = img1.copy()
    temp[0:rotated.shape[0], 0:rotated.shape[1]] = cv2.bitwise_xor(img1[0:rotated.shape[0], 0:rotated.shape[1]],
                                                                   rotated)
    cv2.imwrite(n + '4.jpg', temp)


def hOCR_detecting_lines(m, s):
    img = cv2.imread(str(m) + str(s) + '.jpg')
    f = pytesseract.pytesseract.image_to_pdf_or_hocr(img, lang='fas+ara', extension='hocr')
    tree = etree.fromstring(f)
    words = tree.xpath("//*[@class='ocr_line']")
    for w in words:
        title_splited = w.attrib['title'].split()
        x1, y1, x2, y2 = int(title_splited[1]), int(title_splited[2]), int(title_splited[3]), int(
            title_splited[4].split(';')[0])
        img_hocr = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)


#    cv2.imwrite(str(m) + str(s) + 'rec.jpg', img_hocr)


def hough_detecting_lines(img_sent, m=0, s=0, source_img=0, slice=10):
    if not img_sent:
        img = cv2.imread(str(m) + str(s) + '.jpg')
    else:
        img = source_img

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass
    edges = cv2.Canny(img, 50, 200, apertureSize=3)

    minLineLength = 15
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength, maxLineGap)
    print (lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 2)


#    cv2.imwrite(str(m) + str(s) + '_hough_line.jpg', img)


def pixel_counting_detecting(m, s, erod_itter, dilate_itter, ker_num, th, bias, main_border_percent):
    img = cv2.imread(str(m) + str(s) + '.jpg')
    img_file = str(m) + str(s) + '.jpg'
    print (img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1 - we need dpi for slicing image
    imgPIL = Image.open(img_file)
    dpi = (300, 300)  # default is (300 , 300)
    if 'dpi' in imgPIL.info.keys():
        dpi = imgPIL.info['dpi']
    del imgPIL

    # 2 - use erod nad then dilate in order to clear small noises
    gray_env = cv2.bitwise_not(gray)

    kernel_dilate = np.ones((ker_num, ker_num), np.uint8)
    gray_env_erod = cv2.erode(gray_env, kernel_dilate, iterations=erod_itter)
    gray_env_dilate = cv2.dilate(gray_env_erod, kernel_dilate, iterations=dilate_itter)

    # 3 - by semi-histogram way we want to find wasted areas
    slice = int(dpi[0] / 20)
    # cv2.imwrite('find_wasted_round_area_in_documents_1_inv.jpg', gray_env_dilate)

    poly = np.zeros((int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice), 1), np.uint8)
    poly.fill(0)
    pices = (int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice))
    for y in range(pices[0]):
        for x in range(pices[1]):
            poly[y, x] = np.mean(gray_env_dilate[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)])
    _, poly = cv2.threshold(poly, 10, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('find_wasted_round_area_in_documents_2_poly_1.jpg', poly)

    poly2 = np.zeros((int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice), 1), np.uint8)
    poly2.fill(0)
    min_avg_picx = ((th - 1) * 255) / (th ** 2) - bias
    print ('min_avg_picx', min_avg_picx)
    for y in range(th, pices[0] - th):
        for x in range(th, pices[1] - th):
            if (np.mean(poly[y - th:y + th + 1, x - th:x + th + 1]) > min_avg_picx):
                poly2[y - th:y + th + 1, x - th:x + th + 1] = 255
            else:
                poly2[y, x] = 0

    # cv2.imwrite('find_wasted_round_area_in_documents_4_poly2_{}_{}.jpg'.format(str(m),str(s)), poly2)

    del poly
    poly3 = np.zeros((int(gray_env_dilate.shape[0]), int(gray_env_dilate.shape[1]), 1), np.uint8)
    poly3.fill(0)
    for y in range(0, pices[0]):
        for x in range(0, pices[1]):
            poly3[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)] = poly2[y, x]

    # cv2.imwrite(img_file+'find_wasted_round_area_in_documents_5_poly3.jpg', poly3)
    del poly2

    contours, _ = cv2.findContours(poly3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    poly3 = cv2.drawContours(poly3, contours, -1, (127, 127, 127), 100)

    destination_of_main_paragraphs = img.copy()
    destination_of_main_paragraphs.fill(255)

    rect_main_paragraphs = img.copy()
    borders = []
    for cnt in contours[:]:

        x, y, w, h = cv2.boundingRect(cnt)
        box = np.int0([[x + w, y + h], [x, y + h], [x, y], [x + w, y]])
        poly3 = cv2.drawContours(poly3, [box], 0, (200, 200, 200), 30)

        first_sorted = sorted(box, key=lambda l: l[0])
        lefts = first_sorted[0:2]
        rights = first_sorted[2:]
        tmp = sorted(lefts, key=lambda l: l[1])
        top_left = tmp[0]
        down_left = tmp[1]

        tmp = sorted(rights, key=lambda l: l[1])
        top_right = tmp[0]
        down_right = tmp[1]

        h_half, w_half = img.shape[0] / 2, img.shape[1] / 2

        h_border_top, h_border_down, w_border_left, w_border_right = \
            img.shape[0] * main_border_percent[0], img.shape[0] * (1 - main_border_percent[1]), \
            img.shape[1] * main_border_percent[1], img.shape[1] * (1 - main_border_percent[1])

        if (((min(top_left[1], top_right[1]) < h_half and max(down_left[1], down_right[1]) > h_half) and
             (min(down_left[0], top_left[0]) < w_half and max(top_right[0], down_right[0]) > w_half)) or
                w_border_left < top_left[0] and w_border_left < down_left[0] and
                top_right[0] < w_border_right and down_right[0] < w_border_right and
                h_border_top < top_left[1] and h_border_top < top_right[1] and
                down_right[1] < h_border_down and down_left[1] < h_border_down):
            # print((top_left , top_right),  'main paragraph')
            destination_of_main_paragraphs = cv2.drawContours(destination_of_main_paragraphs, [cnt], 0, (0, 0, 0), -1)
            rect_main_paragraphs = cv2.drawContours(rect_main_paragraphs, [box], 0, (0, 0, 0), 10)

            x, y, w, h = cv2.boundingRect(cnt)
            borders.append([x, y, x + w, y + h])

        else:
            c = 1

            if (((top_left[1] - down_left[1]) ** 2 + (top_left[0] - down_left[0]) ** 2) <
                    ((top_left[1] - top_right[1]) ** 2 + (top_left[0] - top_right[0]) ** 2)):

                # print ('horosontal',c)

                y1 = down_left[0]
                x1 = down_left[1]
                y2 = down_right[0]
                x2 = down_right[1]
                angle = (x2 - x1) / (y1 - y2)
                degree = (np.arctan(angle) / np.pi) * 180
                # print(x1 , y1 , x2 , y2)
                # print('angle: ',(np.arctan(angle)/np.pi)*180)

            else:
                # print ('vertical')
                y1 = down_left[0]
                x1 = down_left[1]
                y2 = top_left[0]
                x2 = top_left[1]
                if y1 != y2:
                    angle = (x2 - x1) / (y1 - y2)
                else:
                    angle = 90
                degree = (np.arctan(angle) / np.pi) * 180
                # print(x1 , y1 , x2 , y2)
                # print('angle: ', (np.arctan(angle) / np.pi) * 180)

            # img = cv2.drawContours(img,[box],0,(1*c,2*c,3*c),5)

            cv2.putText(img, str(degree), (down_right[0], down_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            # print (degree)
            if degree > 5:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < img.shape[1] * .5 and h < img.shape[0] * .5:
                    # print(c,' must be changed' , ' => ',w,h)
                    new_img = img[y:y + h, x:x + w]
                    rotated = ndimage.rotate(new_img, -1 * degree)
                    cv2.floodFill(rotated, None, (0, 0), (255, 255, 255))
                    cv2.imwrite('{}_{}_{}_over_rotated_paragraph.jpg'.format(str(m), str(s), str(c)), rotated)
            c += 1

    # boider detection
    border_of_document = cv2.rectangle(img,
                                       (min(borders, key=lambda x: x[0])[0], min(borders, key=lambda x: x[1])[1]),
                                       (max(borders, key=lambda x: x[2])[2], max(borders, key=lambda x: x[3])[3]),
                                       (0, 0, 0), 3)
    cv2.imwrite('{}_{}_{}_border_of_document.jpg'.format(str(m), str(s)), border_of_document)
    cv2.imwrite('{}_{}_{}_find_wasted_round_area_in_documents_5_poly3.jpg'.format(str(m), str(s)), poly3)
    clear_dest = cv2.bitwise_or(img, destination_of_main_paragraphs)
    cv2.imwrite('{}_{}_{}_main_paragrapha.jpg'.format(str(m), str(s), str(c)), clear_dest)
    cv2.imwrite('{}_{}_{}_rectangled_main_paragrapha.jpg'.format(str(m), str(s)), rect_main_paragraphs)


if __name__ == '__main__':
    M = 31
    S = 1

    avg_time = []
    for m in range(31, M + 1):
        for s in range(1, 1 + S):
            e1 = cv2.getTickCount()

            # per any type of documents set this things:
            erod_itter = 3
            dilate_itter = 1  # same for most of them
            ker_num = 2  # same for most of them
            th = 4
            bias = 30

            # percents of main border in documents
            main_border_percent = [0.06, 0.05]  # a paragraph from h in %6 from left till %100-6 is in main paragraphs,
            #                                     a paragraph from w in %5 from left till %100-5 is in main paragraphs,

            pixel_counting_detecting(m, s, erod_itter, dilate_itter, ker_num, th, bias, main_border_percent)
            e2 = cv2.getTickCount()
            print(m, s)
            avg_time.append((e2 - e1) / cv2.getTickFrequency())
    print(np.mean(avg_time))
