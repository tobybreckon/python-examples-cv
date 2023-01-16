#####################################################################

# Example : HOG pedestrain detection from a video file
# specified on the command line (e.g. FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
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

# if we have OpenCL H/W acceleration availale, use it - we'll need it

cv2.ocl.setUseOpenCL(True)
print(
    "INFO: OpenCL - available: ",
    cv2.ocl.haveOpenCL(),
    " using: ",
    cv2.ocl.useOpenCL())

#####################################################################


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the
        # real objects so we slightly shrink the rectangles to
        # get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h),
                      (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

#####################################################################

# power law transform
# image - colour image
# gamma - "gradient" co-efficient of gamma function


def powerlaw_transform(image, gamma):

    # compute power-law transform
    # remembering not defined for pixel = 0 (!)

    # handle any overflow in a quick and dirty way using 0-255 clipping

    image = np.clip(np.power(image, gamma), 0, 255).astype('uint8')

    return image


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
        cap = camera_stream.CameraVideoStream()  # T-API done later
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

window_name = "HOG pedestrain detection"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # set up HoG detector

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # add some track bar controllers for settings

    neighbourhood = 3
    cv2.createTrackbar("Smoothing : neighbourhood, N", window_name,
                       neighbourhood, 40, nothing)

    sigma = 1
    cv2.createTrackbar("Smoothing : sigma", window_name, sigma, 10, nothing)

    gamma = 100  # default gamma = 100 * 0.01 = 1 -> no change
    cv2.createTrackbar("gamma, (* 0.01)", window_name, gamma, 150, nothing)

    svm_threshold = 0  # by default the SVM's own threshold at the hyperplane
    cv2.createTrackbar("SVM threshold, (distance from hyper-plane, * 0.1)",
                       window_name, svm_threshold, 10, nothing)

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
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # get parameters from track bars

        neighbourhood = cv2.getTrackbarPos(
            "Smoothing : neighbourhood, N", window_name)
        sigma = cv2.getTrackbarPos("Smoothing : sigma", window_name)
        gamma = cv2.getTrackbarPos("gamma, (* 0.01)", window_name) * 0.01
        svm_threshold = cv2.getTrackbarPos(
            "SVM threshold, (distance from hyper-plane, * 0.1)",
            window_name) * 0.1

        # check neighbourhood is greater than 3 and odd

        neighbourhood = max(3, neighbourhood)
        if not (neighbourhood % 2):
            neighbourhood = neighbourhood + 1

        # use power-law function to perform gamma correction
        # and convert np array to T-API universal array for H/W acceleration

        frame = cv2.UMat(powerlaw_transform(frame, gamma))

        # perform Gaussian smoothing using NxN neighbourhood

        frame = cv2.GaussianBlur(
            frame,
            (neighbourhood,
             neighbourhood),
            sigma,
            sigma,
            borderType=cv2.BORDER_REPLICATE)

        # perform HOG based pedestrain detection

        found, w = hog.detectMultiScale(
            frame, winStride=(
                8, 8), padding=(
                32, 32), scale=1.05, hitThreshold=svm_threshold)
        found_filtered = []

        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
                else:
                    found_filtered.append(r)

        draw_detections(frame, found_filtered, 3)

        # display image

        cv2.imshow(window_name, frame)

        # stop the timer and convert to ms. (to see how long processing and
        # display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # e.g. if user presses "x" then exit  / press "f" for fullscreen
        # display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")
