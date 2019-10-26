################################################################################

# Example : perform live chromaticity/lightness display from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon, Engineering & Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

################################################################################

import cv2
import argparse
import sys
import math
import numpy as np

################################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

################################################################################

# concatenate two RGB/grayscale images horizontally (left to right) handling
# differing channel numbers or image heights in the input

def h_concatenate(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0]
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
    else: # both must be equal
        out1 = img1
        out2 = img2

    # height of first image is master height, width can remain unchanged

    if (height1 != height2):
        out2 = cv2.resize(out2, (height1, width2))

    return np.hstack((out1, out2))

################################################################################

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

windowName = "Live - [Original RGB | Chromaticity {r,g,b} | Lightness (l)]"

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

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
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)


        # compute chromaticity as  c = c / SUM(RGB) for c = {R, G, B} with
        # safety for divide by zero errors
        # chromaticity {r,g,b} range is floating point 0 -> 1

        # N.B. if extracting chromaticity {r,g} from this remember to
        # take channels r = 2 and g = 1 due to OpenCV BGR channel ordering

        chromaticity = np.zeros(frame.shape).astype(np.float32)
        sum_channel = np.zeros(frame.shape).astype(np.float32)
        sum_channel = (frame[:,:,0].astype(np.float32)
                        + frame[:,:,1].astype(np.float32)
                        + frame[:,:,2].astype(np.float32)) + np.finfo(np.float32).resolution
        chromaticity[:,:,0] = (frame[:,:,0] / sum_channel)
        chromaticity[:,:,1] = (frame[:,:,1] / sum_channel)
        chromaticity[:,:,2] = (frame[:,:,2] / sum_channel)

        # compute lightness as an integer = RGB / 3 (range is 0 -> 255)

        lightness = np.floor(sum_channel / 3)

        # display image as a concatenated triple of [ RGB | Chromaticity | Lightness ]
        # adjusting back to 8-bit and scaling appropriately for display

        cv2.imshow(windowName, h_concatenate(h_concatenate(frame, (chromaticity * 255).astype(np.uint8)), lightness.astype(np.uint8)))
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN & args.fullscreen)

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
            args.fullscreen = not(args.fullscreen)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

################################################################################
