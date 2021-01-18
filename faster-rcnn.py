##########################################################################

# Example : performs Faster R-CNN object detection from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2019 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Implements the Faster R-CNN object detection architecture decribed in full in:
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
# Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun -
# https://arxiv.org/abs/1506.01497

# This code: significant portions based in part on the example available at:
# https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py

# To use first download and unpack the following files:
# https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_coco.txt
# https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt
# http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
# then unpack and rename as follows:
# tar -xzvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
# mv faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
# faster_rcnn_inception_v2_coco_2018_01_28.pb

##########################################################################

import cv2
import argparse
import sys
import math
import numpy as np

##########################################################################

keep_processing = True

# parse command line arguments for camera ID or video file, and Faster
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
    default='faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')
parser.add_argument(
    "-w",
    "--weights_file",
    type=str,
    help="network weights",
    default='faster_rcnn_inception_v2_coco_2018_01_28.pb')

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

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from Faster R-CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression


def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result[0, 0]:
            confidence = detection[2]
            if confidence > confThreshold:
                left = int(detection[3])
                top = int(detection[4])
                right = int(detection[5])
                bottom = int(detection[6])
                width = right - left + 1
                height = bottom - top + 1
                if width <= 2 or height <= 2:
                    left = int(detection[3] * frameWidth)
                    top = int(detection[4] * frameHeight)
                    right = int(detection[5] * frameWidth)
                    bottom = int(detection[6] * frameHeight)
                    width = right - left + 1
                    height = bottom - top + 1
                # no background label for F-RCNN
                classIds.append(int(detection[1]))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        threshold_confidence,
        threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

##########################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

# init Faster R-CNN object detection model

nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 600       # Width of network's input image
inpHeight = 800      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network
# using them

net = cv2.dnn.readNet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib
# available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

# change to cv2.dnn.DNN_TARGET_CPU or cv2.dnn.DNN_TARGET_OPENCL (slower)
# if this causes issues (should fail gracefully if CUDA/OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

##########################################################################

# define display window name + trackbar

window_name = 'Faster R-CNN object detection: ' + args.weights_file
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

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels not
        # scaled, image resized)
        tensor = cv2.dnn.blobFromImage(
            frame, 1.0, (inpWidth, inpHeight), [
                0, 0, 0], swapRB=False, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName, window_name) / 100
        classIDs, confidences, boxes = postprocess(
            frame, results, confThreshold, nmsThreshold)

        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(frame,
                     classes[classIDs[detected_object]],
                     confidences[detected_object],
                     left,
                     top,
                     left + width,
                     top + height,
                     (150,
                      178,
                      50))

        # stop the timer and convert to ms. (to see how long processing takes

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
