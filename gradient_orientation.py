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

parser = argparse.ArgumentParser(
    description='Perform ' +
    sys.argv[0] +
    ' example operation on incoming camera/video image')
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-r",
    "--rescale",
    type=float,
    help="rescale image by this factor",
    default=1.0)
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
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

    if not (args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window names

window_nameGx = "Gradient - Gx"  # window name
window_nameGy = "Gradient - Gy"  # window name
window_nameAngle = "Gradient Angle"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_nameGx, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameGy, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameAngle, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings

    lower_threshold = 0
    cv2.createTrackbar(
        "lower",
        window_nameAngle,
        lower_threshold,
        180,
        nothing)

    upper_threshold = 180
    cv2.createTrackbar(
        "upper",
        window_nameAngle,
        upper_threshold,
        180,
        nothing)

    neighbourhood = 3
    cv2.createTrackbar(
        "neighbourhood, N",
        window_nameGy,
        neighbourhood,
        40,
        nothing)

    sigma = 1
    cv2.createTrackbar(
        "sigma",
        window_nameGy,
        sigma,
        10,
        nothing)

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
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # get parameter from track bars - Gaussian pre-smoothing

        neighbourhood = cv2.getTrackbarPos("neighbourhood, N", window_nameGy)
        sigma = cv2.getTrackbarPos("sigma", window_nameGy)

        # check neighbourhood is greater than 3 and odd

        neighbourhood = max(3, neighbourhood)
        if not (neighbourhood % 2):
            neighbourhood = neighbourhood + 1

        # perform Gaussian smoothing using NxN neighbourhood

        smoothed_img = cv2.GaussianBlur(
            frame,
            (neighbourhood,
             neighbourhood),
            sigma,
            sigma,
            borderType=cv2.BORDER_REPLICATE)

        # compute the gradients in the x and y directions separately
        # N.B from here onward these images are 32-bit float

        gx = cv2.Sobel(smoothed_img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(smoothed_img, cv2.CV_32F, 0, 1)

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

        # get threshold from trackbars and threshold to keep inner range

        lower_threshold = cv2.getTrackbarPos("lower", window_nameAngle)
        upper_threshold = cv2.getTrackbarPos("upper", window_nameAngle)

        mask = cv2.inRange(angle, lower_threshold, upper_threshold)
        angle = cv2.bitwise_and(angle.astype(np.uint8), mask)

        # display images (as 8-bit)

        cv2.imshow(window_nameGx, gx.astype(np.uint8))
        cv2.imshow(window_nameGy, gy.astype(np.uint8))
        cv2.imshow(window_nameAngle, angle.astype(np.uint8))

        # stop the timer and convert to ms. (to see how long processing and
        # display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # milliseconds). It waits for specified milliseconds for any keyboard
        # event. If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response)

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen
        # display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_nameAngle,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
