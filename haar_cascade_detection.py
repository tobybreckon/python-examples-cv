#####################################################################

# Example : perform haar cascade detection on live display from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based on example at:
# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

# get trained cascade files from:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

#####################################################################

import cv2
import argparse
import sys
import os
import math

#####################################################################

keep_processing = True
faces_recorded = 0

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-ha", "--harvest",  type=str, help="path to save detected faces to", default='')
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

#####################################################################
# set up directory to save faces to if specified

if (len(args.harvest) > 0):
    try:
        os.mkdir(args.harvest)
    except OSError:
        print("Harvesting to existing directory: " + args.harvest)

#####################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture() # not needed for video files

except:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

windowName = "Face Detection using Haar Cascades" # window name

# define haar cascade objects

# required cascade classifier files (and many others) available from:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if (face_cascade.empty() or eye_cascade.empty()):
    print("Failed to load cascade from file.")

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces using haar cascade trained on faces

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30,30), flags=cv2.CASCADE_DO_CANNY_PRUNING)

        # for each detected face, try to detect eyes inside the top
        # half of the face region face region

        for (x,y,w,h) in faces:

            # extract regions of interest (roi) and draw each face bounding box and

            roi_gray = gray[y:y+math.floor(h * 0.5), x:x+w] # top 50% to detect eyes
            roi_color = frame[y:y+h, x:x+w].copy() # copy to save if required

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # detect eyes using haar cascade trained on eyes

            eyes = eye_cascade.detectMultiScale(roi_gray)

            # for each detected eye, draw bounding box

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(frame,(x + ex,y + ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

            # if specified, record all the faces we see to a specified directory

            if (len(args.harvest) > 0):
                filename = os.path.join(args.harvest, "face_" + str(format(faces_recorded,'04')) + ".png")
                cv2.imwrite(filename, roi_color)
                faces_recorded += 1

        # display image

        cv2.imshow(windowName,frame)

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
        # here we use a wait time in ms. that takes account of processing time already used in the loop

        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")
