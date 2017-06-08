"""This is the main program to find the text"""

#from imutils import paths
#import argparse

import os
import sys
import time
import re
import argparse
from PIL import Image
import numpy as np
import cv2
import pytesseract
# pylint: disable= E1101


INDEX = 0
j = 0
OUT = 0
SHOW = ""
SAVE = ""


CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17,
    'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26,
    'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35,
    'X': 36, 'Y': 37, 'Z': 38
}

VIDEO_FILE_EXTENSIONS = ('.mp4', '.avi')

IMAGE_FILE_EXTENSIONS = ('.JPG', '.jpg')



def is_video_file(filename):
    """Finding weather it's video or not"""
    return filename.endswith(VIDEO_FILE_EXTENSIONS)


def is_image_file(filename):
    """Finding weather it's video or not"""
    return filename.endswith(IMAGE_FILE_EXTENSIONS)


def main_video(cap):
    """The Main Method"""
    cnt = 0
    start_time = time.clock()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cnt += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blur = variance_of_laplacian(gray)
            if blur > 50:
                # For white vertical text
                white(gray, frame)
                black(gray, frame)
            frame = cv2.resize(frame, (1000, 500))
        k = cv2.waitKey(5) & 0xff
        if k == 27 or ret is False:
            print (time.clock() - start_time, "seconds", cnt)
            #sys.exit(1)
            return
        if SHOW == "video":
            cv2.imshow("Result Window", frame)
    cap.release()
    cv2.destroyAllWindows()

def main_image(cap):
    """The Main Method"""
    cnt = 0
    start_time = time.clock()
    frame = cv2.imread(cap)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm > 50:
        white(gray,frame)
        black(gray,frame)
    frame = cv2.resize(frame, (1000, 500))
    print (time.clock() - start_time, "seconds")
    if SHOW == "image":
        cv2.imshow(cap, frame)

    #cv2.destroyAllWindows()


def variance_of_laplacian(image):
    """Finding Image Blur value"""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def white(gray, frame):
    """Finding White text process"""
    mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)[1]
    image_final = cv2.bitwise_and(gray, gray, mask=mask)
    new_img = cv2.threshold(image_final, 230, 255, cv2.THRESH_BINARY)[1]
    if SHOW == "white":
        cv2.imshow("white", new_img)
    horizantal(new_img, frame, "white")
    vertical(new_img, frame, "white")

    


def black(gray, frame):
    """Finding Black text process"""
    mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    image_final = cv2.bitwise_and(gray, gray, mask=mask)
    new_img = cv2.threshold(image_final, 30, 255, cv2.THRESH_BINARY_INV)[1]
    if SHOW == "black":
        cv2.imshow("black", new_img)
    horizantal(new_img, frame, "black")
    vertical(new_img, frame, "black")



def vertical(new_img, frame, color):
    """Finding vertical strip text process"""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 13))
    dilated = cv2.dilate(new_img, kernel, iterations=2)

    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1] # get contours

##    for contour in contours:
##        [x_axis, y_axis, width, height] = cv2.boundingRect(contour)
##        cv2.rectangle(frame, (x_axis, y_axis), (x_axis+width, y_axis+height),
##                      (255, 0, 255), 2)
##        cv2.putText(frame,str(x_axis)+'_'+str(y_axis)+'_'+str(width)+'_'+str(height), (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    if SHOW == "vertical":
        dilated = cv2.resize(dilated, (650, 400))
        cv2.imshow(color + " vertical", dilated)
##    cv2.waitKey(0)

        
    for contour in contours:
        [x_axis, y_axis, width, height] = cv2.boundingRect(contour)
        #print x_axis, y_axis, width, height
        if (5 < width < 100) and (100 < height ):
            if (height/width) > 6 and (height/width) < 30:
                save_cropped_image(x_axis, y_axis, width, height, new_img, frame, "vertical")


def horizantal(new_img, frame, color):
    """Finding horizantal stip text process"""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    dilated = cv2.dilate(new_img, kernel, iterations=3)
    if SHOW == "horizantal":
        dilated = cv2.resize(dilated, (650, 400))
        cv2.imshow(color+" black", dilated)
##    cv2.waitKey(0)

    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    for contour in contours:
        [x_axis, y_axis, width, height] = cv2.boundingRect(contour)
        if (20 < height < 350) and (300 < width < 1500):
            if (width/height) > 4  and (width/height) < 20:
                save_cropped_image(x_axis, y_axis, width, height, new_img, frame, "horizantal")


def save_cropped_image(x_axis, y_axis, width, height, new_img, frame, strip_type):
    """vertical and horizantal images saving"""
    global INDEX

##    cv2.rectangle(frame, (x_axis, y_axis), (x_axis+width, y_axis+height),
##                  (255, 0, 255), 2)
    cropped = new_img[y_axis : y_axis +  height, x_axis : x_axis + width]
    cropped = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(cropped, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    if SAVE == "orginal":
        cv2.imwrite("outputs\\rstrip\\crop_mini_v"+str(INDEX)+".jpg",
                frame[y_axis : y_axis +  height, x_axis : x_axis + width])
        INDEX = INDEX + 1
    get_text(dilation, strip_type, frame, x_axis, y_axis, width, height)


def get_text(dilation, angle, frame, strip_x_axis, strip_y_axis, strip_width, strip_height):
    """Getting text from image"""
    global j, OUT
    new_img = cv2.threshold(dilation, 180, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    new_img = cv2.dilate(new_img, kernel, iterations=1)
    contours = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    count = 0
    #print len(contours)
    for contour in contours:
        count += 1
##        [x_axis, y_axis, width, height] = cv2.boundingRect(contour)
##        if width < height:
##            count += 1
##    print count
    if count == 12 or count == 11 or count == 15:
        temp = []
        tempo = []
        for contour in contours:
            [x_axis, y_axis, width, height] = cv2.boundingRect(contour)
            if width < height and (float(height)/float(width)) > 1:
                tempo.append([x_axis, y_axis, width, height])
        if angle == "horizantal":
            tempoo = sorted(tempo)
        else:
            tempoo = reversed(tempo)
        for sub in tempoo:
            x_axis, y_axis, width, height = sub
            cropped = dilation[y_axis : y_axis +  height, x_axis : x_axis + width]
            bordersize = 5
            cropped = cv2.resize(cropped, (20, 30), interpolation=cv2.INTER_CUBIC)
            cropped = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)[1]
            cropped = cv2.copyMakeBorder(cropped, top=bordersize,
                                         bottom=bordersize,
                                         left=bordersize,
                                         right=bordersize,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=[255, 255, 255])
##                s = 'rfstrip/crop_' +str(j)+ '.jpg'
##                cv2.imwrite(s, cropped)
            temp.append(cropped)
##                j += 1
            temp2 = []
##                for i in reversed(temp):
##                    temp2.append(i)
            if temp:
                if len(temp) == 11:
                    cv2.rectangle(frame, (strip_x_axis, strip_y_axis),
                                  (strip_x_axis+strip_width, strip_y_axis+strip_height),
                                  (255, 0, 255), 2)

                    icount = 0
                    for i in temp:
                        kernel = np.ones((2, 2), np.uint8)
                        if icount == 10:
                            i = cv2.dilate(i, kernel, iterations=2)
                            i = cv2.erode(i, kernel, iterations=2)
                        if SAVE == "single":
                            cv2.imwrite('outputs\\rfstrip\\crop_' +str(j)+'.jpg', i)
                            j += 1
                        icount += 1
                        temp2.append(i)
                    if SAVE == "generated":
                        cv2.imwrite('outputs\\routstrip\\strip'+str(OUT)+'.jpg',
                                np.concatenate(temp2, axis=1))
                    else:
                        cv2.imwrite('outputs\\routstrip\\strip0.jpg',
                                cv2.blur(np.concatenate(temp2, axis=1),(5,5)))
                        OUT = 0
                    container_text = get_output(OUT)
                    if container_text:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,container_text ,(strip_x_axis,strip_y_axis), font, 1,(0, 0, 0),2,cv2.LINE_AA)
                    OUT += 1


def get_output(output):
    """reading strip image and convert into text"""
    bw_img = Image.open('outputs\\routstrip\\strip'+str(output)+'.jpg')
    
    txt = pytesseract.image_to_string((bw_img), lang='eng')
    txt = txt.upper()
    #print angle
    txt = re.sub('[^A-Z0-9]', '', txt)
    txt1 = txt[:4].replace(' ', '')
    txt2 = txt[4:].replace('A', '4')
    txt2 = txt2.replace('B', '6')
    txt2 = txt2.replace('I', '1')
    txt2 = txt2.replace('L', '1')
    txt2 = txt2.replace('O', '0')
    txt2 = txt2.replace('S', '5')
    txt2 = txt2.replace('T', '7')
    txt2 = txt2.replace('Z', '2')
    txt2 = txt2.replace(' ', '')
    txt = txt1+txt2
    #.replace('','')
##    print txt
    if len(txt) == 11 and re.search('([A-Z]{4}[0-9]{7})', txt) != None:
        cid = txt
        first10 = cid[0:-1]
        check = cid[-1]
        total = sum(CHARACTERS[c] * 2**x for x, c in enumerate(first10))
        print txt
        if((total % 11) % 10) == CHARACTERS[check]:
            print txt[:4], txt[4:10], txt[10:11], '(Right)'
            return txt[:4]+" "+ txt[4:10]+" "+txt[10:11]
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    if not os.path.exists("outputs\\rstrip"):
        os.makedirs("outputs\\rstrip")
    if not os.path.exists("outputs\\rfstrip"):
        os.makedirs("outputs\\rfstrip")
    if not os.path.exists("outputs\\routstrip"):
        os.makedirs("outputs\\routstrip")



    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-i", "--image", help="For Single Image")
    PARSER.add_argument("-im", "--imagemulti", nargs="+", help="For Multiple Images")
    PARSER.add_argument("-if", "--imagefolder", help="For All Images in a Folder")
    PARSER.add_argument("-v", "--video", help="For Single Video")
    PARSER.add_argument("-vm", "--videomulti", nargs="+", help="For Multiple Videos")
    PARSER.add_argument("-vf", "--videofolder", help="For All Videos in a Folder")
    PARSER.add_argument("-sh", "--showimage", help="For Display the Video or Image")
    PARSER.add_argument("-s", "--save", help="To Save Right Images, Individual and Strip Images")
    PARSER.add_argument("-ims", "--imagestring", help="To get the text from image directly")

    ARGS = PARSER.parse_args()
    if ARGS.showimage:
        SHOW = ARGS.showimage

    if ARGS.save:
        SAVE = ARGS.save

    if ARGS.image:
        print "single image", ARGS.image
        if os.path.exists(ARGS.image):
            main_image(ARGS.image)

        else:
            print "Image "+ARGS.image+" Not Available in this Location"
        cv2.waitKey(0)


    elif ARGS.imagemulti:
        for multi in ARGS.imagemulti:
            print "multiple images", multi
            if os.path.exists(multi):
                main_image(multi)

            else:
                print "Image "+multi+" Not Available in this Location"
        cv2.waitKey(0)

    elif ARGS.imagefolder:
        print "folder images", ARGS.imagefolder
        for multi in os.listdir(ARGS.imagefolder):
            if is_image_file(multi):
                main_image(ARGS.imagefolder+"/"+multi)
            else:
                print ""+multi+" Not a image file in this Location"
        cv2.waitKey(0)

    elif ARGS.video:
        if os.path.exists(ARGS.video):
            main_video(cv2.VideoCapture(ARGS.video))
        else:
            print "Video "+ARGS.video+" Not Available in this Location"

    elif ARGS.videomulti:
        for multi in ARGS.videomulti:
            if os.path.exists(multi):
                main_video(cv2.VideoCapture(multi))
                #print i
            else:
                print "Video "+multi+" Not Available in this Location"

    elif ARGS.videofolder:
        print "folder videos", ARGS.videofolder
        for multi in os.listdir(ARGS.videofolder):
            if is_video_file(multi):
                main_video(cv2.VideoCapture(ARGS.videofolder+"/"+multi))
            else:
                print ""+multi+" Not a Video file in this Location"

    elif ARGS.imagestring:
        if os.path.exists(ARGS.imagestring):
                img = Image.open(ARGS.imagestring)
                txt = pytesseract.image_to_string((img))
                print txt

        else:
            print "Image "+ARGS.imagestring+" Not Available in this Location"
        cv2.waitKey(0)

    
    else:
        main_video(cv2.VideoCapture('inputs\\testv.mp4'))
        sys.exit(0)
