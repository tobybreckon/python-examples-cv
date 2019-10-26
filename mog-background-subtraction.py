#####################################################################

# Example : perform GMM based foreground/background subtraction from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015-18 Toby Breckon, Engineering & Computer Science,
#                       Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys

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

# check versions to work around this bug in OpenCV 3.1
# https://github.com/opencv/opencv/issues/6055

(major, minor, _) = cv2.__version__.split(".")
if ((major == '3') and (minor == '1')):
    cv2.ocl.setUseOpenCL(False)

# define display window name

windowName = "Live Camera Input" # window name
windowNameBG = "Background Model" # window name
windowNameFG = "Foreground Objects" # window name
windowNameFGP = "Foreground Probabiity" # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL)

    # create GMM background subtraction object
    # (using default parameters which are suitable for quick lecture demos
    # - see manual for suitable choice of values to use in anger)

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

    print("\nPress <space> to reset MoG model ...\n")

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

        # add current frame to background model and retrieve current foreground objects

        fgmask = mog.apply(frame)

        # threshold this and clean it up using dilation with a elliptical mask

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)

        # get current background image (representative of current GMM model)

        bgmodel = mog.getBackgroundImage()

        # display images - input, background and original

        cv2.imshow(windowName,frame)
        cv2.imshow(windowNameFG,fgdilated)
        cv2.imshow(windowNameFGP,fgmask)
        cv2.imshow(windowNameBG, bgmodel)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit or reset MoG modelw when space is presses

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord(' ')):
            print("\nResetting MoG background model ...\n")
            mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
