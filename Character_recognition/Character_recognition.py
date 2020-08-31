import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
import time
import pyttsx3

#import subprocess
#subprocess.call(["say",text])

def say(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def call():
    c1 ='Press 1 to rescan again : '
    say(c1)
    i = int(input(c1))
    if i==1:
        main()
    else:
        c1 ='Quiting...'
        say(c1)
        print(c1)
        
def main():
    c1 ='Capturing Image...'
    say(c1)

    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite('img.jpg',image)
    del(camera)

    c1 ='Capturing done... Processing...  Please wait'
    say(c1)

    img = cv2.imread('img.jpg',cv2.IMREAD_COLOR)

    img = cv2.resize(img, (620,480) )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(gray, 30, 200) 
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in cnts:
     peri = cv2.arcLength(c, True)
     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
     if len(approx) == 4:
      screenCnt = approx
      break

    if screenCnt is not None:
         detected = 1
    else:
        detected = 0
        c1 ='Area not detected... please fix the image and retry'
        say(c1)
        print (c1)
        call()


    if detected == 1:
     cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    #text = pytesseract.image_to_string(Cropped, config='--psm 11')
    text = pytesseract.image_to_string(Cropped, lang='eng')

    if text is not None:
        c1 = "Text detected:"
        print(c1)
        say(c1)
        time.sleep(1)
        print(text)

        cv2.imshow('image',img)
        #cv2.imshow('Cropped',Cropped)
        say(text)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        call()

    else:
        c1 = "No text detected!."
        print(c1)
        say(c1)
        call()
    
   
main()
