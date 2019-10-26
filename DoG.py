#####################################################################

# Example :  Difference of Gaussian (DoG) of a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017-2019 Dept. Engineering & Dept. Computer Science,
#                         Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import numpy as np

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-i", "--is_image", action='store_true', help="specify file is an image, not a video")
parser.add_argument('video_file', metavar='file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

#####################################################################

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass

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

windowName = "Live Camera Input" # window name
windowNameU = "Gaussian  Upper" # window name
windowNameL = "Gaussian  Lower" # window name
windowNameDoG = "DoG" # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameU, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameDoG, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings

    sigmaU = 2 # greater than 7 seems to crash
    cv2.createTrackbar("sigma U", windowNameU, sigmaU, 15, nothing)
    sigmaL = 1 # greater than 7 seems to crash
    cv2.createTrackbar("sigma L", windowNameL, sigmaL, 15, nothing)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # if it is a still image, load that instead

        if (args.is_image):
            frame = cv2.imread(args.video_file, cv2.IMREAD_COLOR)

        # get parameters from track bars

        sigmaU = cv2.getTrackbarPos("sigma U", windowNameU)
        sigmaL = cv2.getTrackbarPos("sigma L", windowNameL)

        # check sigma's are greater than 1

        sigmaU = max(1, sigmaU)
        sigmaL = max(1, sigmaL)

        # check sigma are correct

        if (sigmaL >= sigmaU) and (sigmaU > 1):
            sigmaL = sigmaU - 1
            print("auto-correcting sigmas such that U > L")

        # convert to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # performing smoothing on the image using a smoothing mark (see manual entry for GaussianBlur())
        # specify 0x0 mask size then size is auto-computed from the sigma values

        smoothedU = cv2.GaussianBlur(gray_frame,(0,0),sigmaU)
        smoothedL = cv2.GaussianBlur(gray_frame,(0,0),sigmaL)


        # perform abs_diff() to get DoG

        DoG = cv2.absdiff(smoothedU, smoothedL)

        # auto-scale to full 0 -> 255 range based on max DoG response
        # noting that as both inputs to absdiff() are 0->255,
        # result will be within range 0->255

        DoG = DoG * (np.max(DoG) / 255)

        # display image

        cv2.imshow(windowName,frame)
        cv2.imshow(windowNameU,smoothedU)
        cv2.imshow(windowNameL,smoothedL)
        cv2.imshow(windowNameDoG,DoG)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        # e.g. if user presses "x" then exit  / press "f" for fullscreen display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowNameDoG, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
