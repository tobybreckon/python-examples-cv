##########################################################################

# Example : perform Gaussian/Laplacian pyramid live display from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2021 Toby Breckon, Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements: based in part from tutorial at:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html

##########################################################################

import cv2
import argparse
import sys
import math
import numpy as np

##########################################################################

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

# define display window name

window_name = "Live Camera Input"  # window name

##########################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # set initial number of pyramid levels

    nlevels = 5

    # print user key commands

    print()
    print("'-' - reduce pyramid levels")
    print("'+' - increase pyramid levels (max 6 levels)")
    print()

    while (keep_processing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # if camera /video file successfully open then read frame

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

        # generate Gaussian pyramid for image frame

        g_level = frame.copy()
        g_pyramid = [g_level]
        for layer in range(nlevels):
            g_level = cv2.pyrDown(g_level)
            cv2.namedWindow("Gaussian Level: " + str(layer),
                            cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Gaussian Level: " + str(layer), g_level)
            g_pyramid.append(g_level.copy())

        # generate Laplacian pyramid image frame

        lp_pyramid = [g_pyramid[nlevels - 1]]
        for layer in range(nlevels, 0, -1):
            g_level_enlarged = cv2.pyrUp(g_pyramid[layer])

            # catch this rounding error occurence in image sizes
            if (g_pyramid[layer-1].shape != g_level_enlarged.shape):
                g_level_enlarged = cv2.resize(
                            g_level_enlarged,
                            tuple(reversed(g_pyramid[layer-1].shape[:2])),
                            interpolation=cv2.INTER_LINEAR)

            l_level = cv2.subtract(g_pyramid[layer-1], g_level_enlarged)
            cv2.normalize(l_level, l_level, 0, 255, cv2.NORM_MINMAX)
            cv2.namedWindow("Laplacian Level: " + str(layer),
                            cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Laplacian Level: " + str(layer), l_level)
            lp_pyramid.append(l_level.copy())

        # display image

        cv2.imshow(window_name, frame)

        # stop the timer and convert to ms. (to see how long processing and
        # display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # start the event loop - essential

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('+')):
            cv2.destroyAllWindows()
            nlevels = np.min([6, nlevels + 1])
        elif (key == ord('-')):
            cv2.destroyAllWindows()
            nlevels = np.max([0, nlevels - 1])

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

##########################################################################
