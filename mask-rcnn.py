##########################################################################

# Example : performs Mask R-CNN object instance segmentation from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2021 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Implements the Mask R-CNN instance segmentation architecture decribed in:
# Mask R-CNN - Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
# https://arxiv.org/abs/1703.06870

# This code: significant portions based in part on the example available at:
# https://github.com/opencv/opencv/blob/master/samples/dnn/segmentation.py

# To use first download and unpack the following files:
# https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_coco.txt
# http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
# https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
# then unpack and rename as follows:
# tar -xzf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

##########################################################################

import cv2
import argparse
import sys
import math
import numpy as np

##########################################################################

keep_processing = True
colors = None

# parse command line arguments for camera ID or video file, and Mask
# R-CNN files
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
    default='cpu')
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
parser.add_argument(
    "-cl",
    "--class_file",
    type=str,
    help="list of classes",
    default='object_detection_classes_coco.txt')
parser.add_argument(
    "-cf",
    "--config_file",
    type=str,
    help="network config",
    default='mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
parser.add_argument(
    "-w",
    "--weights_file",
    type=str,
    help="network weights",
    default="mask_rcnn_inception_v2_coco_2018_01_28/"
            + "/frozen_inference_graph.pb")

args = parser.parse_args()

##########################################################################
# dummy on trackbar callback function


def on_trackbar(val):
    return

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in


def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(
        image,
        (left,
         top -
         round(
             1.5 *
             labelSize[1])),
        (left +
         round(
             1.5 *
             labelSize[0]),
            top +
            baseLine),
        (255,
         255,
         255),
        cv2.FILLED)
    cv2.putText(image, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

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

##########################################################################

# init Mask R-CNN object detection model

inpWidth = 800       # Width of network's input image
inpHeight = 800      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network
# using them

net = cv2.dnn.readNet(args.config_file, args.weights_file)

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

# define display window name + trackbar

window_name = 'Mask R-CNN instance segmentation: ' + args.weights_file
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, window_name, 70, 100, on_trackbar)

##########################################################################

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

        # get frame dimensions
        frameH = frame.shape[0]
        frameW = frame.shape[1]

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels not
        # scaled, image resized)
        tensor = cv2.dnn.blobFromImage(
            frame, 1.0, (inpWidth, inpHeight), [0, 0, 0],
            swapRB=True, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

        # get confidence threshold from trackbar
        confThreshold = cv2.getTrackbarPos(trackbarName, window_name) / 100

        # get number of classes detected and number of detections
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]

        # draw segmentation - first generate colours if needed

        if not colors:
            np.random.seed(324)
            colors = [np.array([0, 0, 0], np.uint8)]
            for i in range(1, numClasses + 1):
                colors.append((colors[i - 1] +
                              np.random.randint(0, 256, [3],
                              np.uint8)) / 2
                              )
            del colors[0]

        # draw segmentation - draw instance segments

        boxesToDraw = []
        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            confidence = box[2]
            if confidence > confThreshold:

                #### draw bounding box (as per Faster R-CNN)

                classId = int(box[1])
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])

                left = max(0, min(left, frameW - 1))
                top = max(0, min(top, frameH - 1))
                right = max(0, min(right, frameW - 1))
                bottom = max(0, min(bottom, frameH - 1))

                drawPred(frame, classes[classId], confidence,
                         left, top, right, bottom, (0,255,0))

                #### draw object instance mask
                # get mask, re-size from 28x28 to size of bounding box
                # then theshold at 0.5

                classMask = mask[classId]
                classMask = cv2.resize(classMask,
                                       (right - left + 1, bottom - top + 1),
                                       cv2.INTER_CUBIC)
                mask = (classMask > 0.5)

                roi = frame[top:bottom+1, left:right+1][mask]
                frame[top:bottom+1, left:right+1][mask] = (
                    0.8 * colors[classId] + 0.2 * roi).astype(np.uint8)

        # stop the timer and convert to ms. (to see how long processing takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # Display efficiency information

        label = ('Inference time: %.2f ms' % stop_t) + \
            (' (Framerate: %.2f fps' % (1000 / stop_t)) + ')'
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & args.fullscreen)

        # start the event loop + detect specific key strokes
        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)
        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # if user presses "x" then exit  / press "f" for fullscreen display
        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            args.fullscreen = not(args.fullscreen)

    # close all windows
    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

##########################################################################
