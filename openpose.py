##########################################################################

# Example : perform live display of openpose body pose regression from a video
# file specified on the command line (e.g. python FILE.py video_file) or from
# an attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2019 Toby Breckon, Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Based heavily on the example provided at:
# https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py

##########################################################################

# To use download COCO model pose files from:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/
# using
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh

##########################################################################

import cv2
import argparse
import sys
import math

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
    "-fs",
    "--fullscreen",
    action='store_true',
    help="run in full screen mode")
parser.add_argument(
    "-use",
    "--target",
    type=str,
    choices=['cpu', 'gpu', 'opencl'],
    help="select computational backend",
    default='gpu')
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

##########################################################################

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

##########################################################################

# define display window name

window_name = "OpenPose Body Pose Regression - Live"  # window name

# create window by name (as resizable)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

##########################################################################

# set pose labels - based on COCO dataset training

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [
                ["Neck", "RShoulder"], ["Neck", "LShoulder"],
                ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"],
                ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"],
                ["LEye", "LEar"]
            ]

##########################################################################

# Load CNN model
net = cv2.dnn.readNet(
    "pose_iter_440000.caffemodel",
    "pose_deploy_linevec.prototxt",
    'caffe')

# set up compute target as one of [GPU, OpenCL, CPU]

if (args.target == 'gpu'):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
elif (args.target == 'opencl'):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

##########################################################################

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

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

        # create a 4D tensor "blob" from a frame - defaults from OpenCV
        # OpenPose example

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=0.003922, size=(
                368, 368), mean=[
                0, 0, 0], swapRB=False, crop=False)

        # Run forward inference on the model

        net.setInput(blob)
        out = net.forward()

        # draw body parts

        if (len(BODY_PARTS) <= out.shape[1]):

            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponding body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums.
                # To simplify a sample we just find a global one.
                # However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]

                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > 0.1 else None)

            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert (partFrom in BODY_PARTS)
                assert (partTo in BODY_PARTS)

                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv2.line(
                        frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(
                        frame, points[idFrom], (3, 3), 0, 0, 360,
                        (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(
                        frame, points[idTo], (3, 3), 0, 0, 360,
                        (0, 0, 255), cv2.FILLED)

        # stop the timer and convert to ms.

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # add efficiency information

        label = ('Inference time: %.2f ms' % stop_t) + \
            (' (Framerate: %.2f fps' % (1000 / stop_t)) + ')'
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        # display image

        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & args.fullscreen)

        # start the event loop - essentials
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
            args.fullscreen = not (args.fullscreen)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

##########################################################################
