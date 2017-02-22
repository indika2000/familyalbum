'''

Local test file for the family album project

'''

from progressbar.progressbar_main import progressbar
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os


path = '/home/indy/Development/familyalbum/familyalbum/'

@progressbar
def vid1():
    """ Show Pic basic function """
    try:
        img = cv2.imread('/home/indy/Development/familyalbum/familyalbum/images/test2.jpg', cv2.IMREAD_GRAYSCALE)
        #cv2 show
        cv2.imshow('img', img)
    except:
        return 1

    #Matplotlib show
    # try:
    #     plt.imshow(img, cmap='gray', interpolation='bicubic')
    #     plt.show()
    # except:
    #     return 1


def vid2():
    """ Show Video Stream - MAX 3 VIDEO screens open at once for my box"""
    cap = cv2.VideoCapture(0)

    #code to capture video and save
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('testvideooutput.avi', fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()

        #You can convert the frame to grey
        #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #test frames
        #test2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        #test1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #test3 = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        #test4 = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)

        cv2.imshow('frame', frame)
        #cv2.imshow('grey', grey)
        #cv2.imshow('test2', test2)

        #cv2.imshow('test1', test1)
        #cv2.imshow('test3', test3)
        #cv2.imshow('test4', test4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()


def vid3():
    """ Drawing on the image"""
    try:
        img = cv2.imread('/home/indy/Development/familyalbum/familyalbum/images/test2.jpg', cv2.IMREAD_COLOR)

        cv2.line(img, (0,0), (150,150), (255, 0, 255), 15)

        cv2.rectangle(img, (15,25), (200, 150), (255, 0, 0), 5)
        cv2.circle(img, (100, 63), 55, (0,0,255), -1)

        #font on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Hello Indy!!', (50, 130), font, 1, (0,255,0))

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass


def vid4():
    """" Video Analysis """
    img = cv2.imread('/home/indy/Development/familyalbum/familyalbum/images/test2.jpg', cv2.IMREAD_COLOR)

    #specify the pixel location - print(px) gives the colour value of that pixel therefore you can amend that pixel value
    px = img[55, 55]

    #This would change the colour of the pixel specified
    img[55, 55] = [255, 255, 255] #RBG colour values in a list to convert the pixel on img at location x55, y55

    #Region of an image
    roi = img[100:150, 100:150] #printing this gives you a map of all the pixel colour values of that region
    img[100:150, 100:150] = [42, 234, 231]

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vid5():
    pass

def vid15():
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('/home/indy/Development/familyalbum/familyalbum/images/people-walking.mp4')

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        cv2.imshow('original', frame)
        cv2.imshow('fg', fgmask)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def vid16():

    face_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.3, 5)

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def vid16_homemade():

    face_cascade = cv2.CascadeClassifier(path + 'data/cascade.xml')

    cap = cv2.VideoCapture(0)


    while True:
        ret, img = cap.read()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 2, 2)

        for(x,y,w,h) in faces:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Hi, Indy", (x-w, y-h), font, 0.5, (0,255,255), 2)
            #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            #roi_grey = grey[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def store_raw_images():
    #n
    ##neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03368352'
    ##neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01905661'
    ##neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n12205694'
    ##neg_images_link = pi/text/imagenet.synset.geturls?wnid=n11883328'
    #neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222'
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01503061'

    try:
        neg_images_urls = urllib.request.urlopen(neg_images_link).read().decode()
    except Exception as e:
        print("moo")

    if not os.path.exists(path+'neg'):
        os.makedirs(path+'neg')


    pic_num = 1399

    for i in neg_images_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, path+'neg/'+str(pic_num)+'.jpg')
            img = cv2.imread(path+'neg/'+str(pic_num)+'.jpg', cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(path+'neg/'+str(pic_num)+'.jpg', resized_image)
            pic_num += 1
        except Exception as e:
            print(str(e))

def face_find():
    try:
        orig = cv2.imread('/home/indy/Development/familyalbum/familyalbum/images/indy.jpg', cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(orig, (0,0), fx=0.25, fy=0.25)

        face_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_grey, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        cv2.imshow('img', img)
    except:
        print("Error")


def vid16_take2():

    face_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path + 'samplehaarcascades/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        funny1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test3 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        test4 = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

        faces = face_cascade.detectMultiScale(grey, 1.3, 5)

        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)


        for(x, y, w, h) in faces:
            cv2.rectangle(funny1, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = funny1[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        for(x, y, w, h) in faces:
            cv2.rectangle(test3, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = test3[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        for(x, y, w, h) in faces:
            cv2.rectangle(test4, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = test4[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_grey)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

        cv2.imshow('img', img)
        cv2.imshow('img2', funny1)
        cv2.imshow('img3', test3)
        cv2.imshow('img4', test4)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def testmeth(ls):
    print(id(ls))
    ls[0] = 100
    print(ls)

    return ls

def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(path+file_type):
            if file_type == 'neg':
                line = path+file_type+'/'+img+'\n'
                with open(path+'/bg.txt', 'a') as f:
                    f.write(line)

            elif file_type == 'pos':
                line = path+file_type+'/'+img+' 1 0 0 50 50\n'
                with open(path + '/info.dat', 'a') as f:
                    f.write(line)
def main():

    #face_find()


    #vid1()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #vid2()

    #vid3()

    #vid4()

    #vid5()
    #vid15()
    #store_raw_images()
    #create_pos_n_neg()

    #vid16_take2()
    #vid16()

    vid16_homemade()

if __name__ == '__main__':
    main()