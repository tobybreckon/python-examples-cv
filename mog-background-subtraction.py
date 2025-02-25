#####################################################################

# Example : perform MoG based foreground/background subtraction from a video
# file specified on the command line (e.g. python FILE.py video_file) or from
# an attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015-25 Toby Breckon, Engineering & Computer Science,
#                       Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import numpy as np

#####################################################################

# concatenate two RGB/grayscale images horizontally (left to right)
# handling differing channel numbers or image heights in the input


def h_concat(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0]
    # width1 = img1.shape[1]
    if (len(img1.shape) == 2):
        channels1 = 1
    else:
        channels1 = img1.shape[2]

    height2 = img2.shape[0]
    width2 = img2.shape[1]
    if (len(img2.shape) == 2):
        channels2 = 1
    else:
        channels2 = img2.shape[2]

    # make all images 3 channel, or assume all same channel

    if ((channels1 > channels2) and (channels1 == 3)):
        out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        out1 = img1
    elif ((channels2 > channels1) and (channels2 == 3)):
        out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        out2 = img2
    else:  # both must be equal
        out1 = img1
        out2 = img2

    # height of first image is master height, width remains unchanged

    if (height1 != height2):
        out2 = cv2.resize(out2, (height1, width2))

    return np.hstack((out1, out2))

#####################################################################

# concatenate two RGB/grayscale images vertically (top to bottom)
# handling differing channel numbers or image heights in the input


def v_concat(img1, img2):

    # get size and channels for both images

    # height1 = img1.shape[0]
    width1 = img1.shape[1]
    if (len(img1.shape) == 2):
        channels1 = 1
    else:
        channels1 = img1.shape[2]

    height2 = img2.shape[0]
    width2 = img2.shape[1]
    if (len(img2.shape) == 2):
        channels2 = 1
    else:
        channels2 = img2.shape[2]

    # make all images 3 channel, or assume all same channel

    if ((channels1 > channels2) and (channels1 == 3)):
        out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        out1 = img1
    elif ((channels2 > channels1) and (channels2 == 3)):
        out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        out2 = img2
    else:  # both must be equal
        out1 = img1
        out2 = img2

    # width of first image is master height, height remains unchanged

    if (width1 != width2):
        out2 = cv2.resize(out2, (height2, width1))

    return np.vstack((out1, out2))

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
    "-s",
    "--set_resolution",
    type=int,
    nargs=2,
    help='override default camera resolution as H W')
parser.add_argument(
    "-fs",
    "--fullscreen",
    action='store_true',
    help="run in full screen mode")
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

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

# check versions to work around this bug in OpenCV 3.1
# https://github.com/opencv/opencv/issues/6055

(major, minor, _) = cv2.__version__.split(".")
if ((major == '3') and (minor == '1')):
    cv2.ocl.setUseOpenCL(False)

# define display window name

window_name = "Live Camera Input"  # window name
window_nameBG = "Background Model"  # window name
window_nameFG = "Foreground Objects"  # window name
window_nameFGP = "Foreground Probabiity"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameBG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameFG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_nameFGP, cv2.WINDOW_NORMAL)

    # override default camera resolution

    if (args.set_resolution is not None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.set_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.set_resolution[0])

    # create GMM background subtraction object
    # (using default parameters which are suitable for quick lecture demos
    # - see manual for suitable choice of values to use in anger)

    mog = cv2.createBackgroundSubtractorMOG2(
        history=2000, varThreshold=16, detectShadows=True)

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
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # add current frame to background model and retrieve current foreground
        # objects (use learningRate parameter for tuning, see manual )

        fgmask = mog.apply(frame)

        # threshold and clean it up using erosion/dilation w/ elliptic mask

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgeroded = cv2.erode(
            fgthres, kernel=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        fgdilated = cv2.dilate(
            fgeroded, kernel=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)

        # get current background image (representative of current GMM model)

        bgmodel = mog.getBackgroundImage()

        # display images - input, background and original

        if (args.fullscreen):

            window_name = "[ Live | BG | Pr(FG) | FG ]"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, v_concat(
                                             h_concat(frame, bgmodel),
                                             h_concat(fgmask, fgeroded)
                                            ))
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN & args.fullscreen)

        else:

            cv2.imshow(window_name, frame)
            cv2.imshow(window_nameFG, fgeroded)
            cv2.imshow(window_nameFGP, fgmask)
            cv2.imshow(window_nameBG, bgmodel)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # ms.) It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response) here we use a wait time in ms. that takes
        # account of processing time already used in the loop

        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit, "f" for fullscreen
        # or reset MoG model when space is pressed

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord(' ')):
            print("\nResetting MoG background model ...\n")
            mog = cv2.createBackgroundSubtractorMOG2(
                history=2000, varThreshold=16, detectShadows=True)
        elif (key == ord('f')):
            args.fullscreen = not (args.fullscreen)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
