from lxml import etree
import cv2
import pytesseract

img = cv2.imread('image1.bmp')
f = pytesseract.pytesseract.image_to_pdf_or_hocr(img , lang='fas+ara',extension='hocr')
tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocrx_word']")
for w in words:
    title_splited = w.attrib['title'].split()
    x1 , y1 , x2 , y2 = int(title_splited[1]) , int(title_splited[2]) , int(title_splited[3]) , int(title_splited[4].split(';')[0])
    img_hocr = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 3)


# gray2 = cv2.cvtColor(img_hocr , cv2.COLOR_BGR2GRAY)
# final = cv2.bitwise_and(new_image , final_img)
cv2.imwrite('rect_words_with_hOCR_EAST_on_denoised_img.jpg',img_hocr)
