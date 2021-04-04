import cv2
import os

def black_white_img(img_url, base_url):
    result = base_url + "/media/images/"
    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filename = os.path.basename(img_url)
    img_name = result + filename

    cv2.imwrite(img_name, img)