#####################################################################

# Example :  canny edge detection for a a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys

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
    "-nc",
    "--nocontrols",
    action='store_true',
    help="no onscreen controls")
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
        cap = camera_stream.CameraVideoStream(use_tapi=True)
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera Input"  # window name
window_name2 = "Canny Edges"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name2, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN & args.fullscreen)

    # add some track bar controllers for settings

    lower_threshold = 25
    upper_threshold = 120
    smoothing_neighbourhood = 3
    sobel_size = 3  # greater than 7 seems to crash

    if (not (args.nocontrols)):
        cv2.createTrackbar("lower", window_name2, lower_threshold,
                           255, nothing)
        cv2.createTrackbar("upper", window_name2, upper_threshold,
                           255, nothing)
        cv2.createTrackbar("smoothing", window_name2, smoothing_neighbourhood,
                           15, nothing)
        cv2.createTrackbar("sobel size", window_name2, sobel_size,
                           7, nothing)

    # override default camera resolution

    if (args.set_resolution is not None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.set_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.set_resolution[0])

    print("INFO: input resolution : (",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), ")")

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()  # rescale if specified

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # get parameters from track bars

        if (not (args.nocontrols)):
            lower_threshold = cv2.getTrackbarPos("lower", window_name2)
            upper_threshold = cv2.getTrackbarPos("upper", window_name2)
            smoothing_neighbourhood = cv2.getTrackbarPos("smoothing",
                                                         window_name2)
            sobel_size = cv2.getTrackbarPos("sobel size", window_name2)

        # check neighbourhood is greater than 3 and odd

        smoothing_neighbourhood = max(3, smoothing_neighbourhood)
        if not (smoothing_neighbourhood % 2):
            smoothing_neighbourhood = smoothing_neighbourhood + 1

        sobel_size = max(3, sobel_size)
        if not (sobel_size % 2):
            sobel_size = sobel_size + 1

        # convert to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # performing smoothing on the image using a 5x5 smoothing mark (see
        # manual entry for GaussianBlur())

        smoothed = cv2.GaussianBlur(
            gray_frame, (smoothing_neighbourhood, smoothing_neighbourhood), 0)

        # perform canny edge detection

        canny = cv2.Canny(
            smoothed,
            lower_threshold,
            upper_threshold,
            apertureSize=sobel_size)

        # display image

        cv2.imshow(window_name, frame)
        cv2.imshow(window_name2, canny)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # milliseconds). It waits for specified milliseconds for any keyboard
        # event. If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response)

        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen
        # display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_name2,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
