from lxml import etree
import cv2
import pytesseract

img = cv2.imread('image1.bmp')
dpi = (300,300) # default is (300,300) but you can check with PIL, Image.info

try:
    f = pytesseract.pytesseract.image_to_pdf_or_hocr(img, lang='fas+ara', extension='hocr')
    # or if you have .hocr file ->f = open('z1.hocr', 'r', encoding='iso-8859-1').read().encode('utf-8')  # 'z1.hocr'
except:
    print('hOCR file was not found')


imgTemp = cv2.imread('image1.bmp') # image file name

tree = etree.fromstring(f)
lines = tree.xpath("//*[@class='ocr_line']")


for line in lines:
    titles = line.attrib['title'].split()
    x1, y1, x2, y2 = int(titles[1]), int(titles[2]), int(titles[3]), int(titles[4].split(';')[0])
    imgTemp = cv2.rectangle(imgTemp , (x1 , y1) , (x2,y2) , (255,0,0) , 3)


cv2.imwrite('rect_lines_with_hOCR.jpg',imgTemp)
