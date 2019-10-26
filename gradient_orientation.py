#####################################################################

# Example : perform generic live display of gradient orientations
# (which form the essensce of the Histogram of Oriented Gradient (HOG) feature)
# from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# https://www.learnopencv.com/histogram-of-oriented-gradients/

# Copyright (c) 2018 Dept. Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import math
import numpy as np

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

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

# define display window names

windowNameGx = "Gradient - Gx" # window name
windowNameGy = "Gradient - Gy" # window name
windowNameAngle = "Gradient Angle" # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowNameGx, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameGy, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameAngle, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

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

        # compute the gradients in the x and y directions separately
        # N.B from here onward these images are 32-bit float

        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

        # calculate gradient magnitude and direction (in degrees)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # normalize

        gx = np.abs(gx)
        gy = np.abs(gy)
        angle = np.abs(angle)

        # normalize other values 0 -> 180

        gx = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX)
        gy = cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX)
        angle = cv2.normalize(angle, None, 0, 180, cv2.NORM_MINMAX)

        # for the angle take the max across all three channels

        (aB, aG, aR) = cv2.split(angle)
        angle = np.maximum(np.maximum(aR, aG), aB)

        # display images (as 8-bit)

        cv2.imshow(windowNameGx,gx.astype(np.uint8))
        cv2.imshow(windowNameGy,gy.astype(np.uint8))
        cv2.imshow(windowNameAngle,angle.astype(np.uint8))

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
            cv2.setWindowProperty(windowNameAngle, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
