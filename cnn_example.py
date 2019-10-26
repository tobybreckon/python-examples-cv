#####################################################################

# Example : perform live object detectoon using a pre-trained CNN model
# and display from a video file specified on the command line
# (e.g. python FILE.py video_file) or from an attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based on provided examples at: https://github.com/opencv/opencv/tree/master/samples/dnn
# see here for how to load Caffe/TensorFlow/... models etc.

# implements a version of:

# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
# Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
# research paper: https://arxiv.org/abs/1704.04861

# requires Caffe network model files (.prototxt / .caffemodel) downloaded from:
# https://github.com/chuanqi305/MobileNet-SSD/

#####################################################################

import cv2
import argparse
import sys
import math
import numpy as np

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

cnn_model_to_load = "MobileNetSSD_deploy"

#####################################################################

def trackbar_callback(pos):
    global confidence_threshold
    confidence_threshold = pos / 100.0

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

# define display window name

windowName = "Live Object Detection - CNN: " + cnn_model_to_load # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    # add track bar to window for confidence threshold

    confidence_threshold = 0.7
    cv2.createTrackbar('Confidence threshold, %', windowName, int(confidence_threshold * 100), 99, trackbar_callback)

    # init CNN model - here from Caffe, although OpenCV can import from
    # mosyt deep learning templates

    net = cv2.dnn.readNetFromCaffe(cnn_model_to_load + ".prototxt", cnn_model_to_load + ".caffemodel")

# provide mappings from class numbers to string labels - these are the PASCAL VOC classees

    classNames = {  0: 'background',
                    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                    14: 'motorbike', 15: 'person', 16: 'pottedplant',
                    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

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
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # get size of input

        cols = frame.shape[1]
        rows = frame.shape[0]

        # transform the image into a network input "blob" (i.e. tensor)
        # by scaling the image to the input size of the network, in this case
        # not swapping the R and G channels (i.e. used when network trained on
        # RGB and not the BGR of OpenCV) and re-scaling the inputs from 0->255
        # to 0->1 by specifing the mean value for each channel

        swapRBchannels = False             # do not swap channels
        crop = False                       # crop image or not
        meanChannelVal = 255.0 / 2.0       # mean channel value

        inWidth = 300                      # network input width
        inHeight = 300                     # network input height
        WHRatio = inWidth / float(inHeight)
        inScaleFactor = 0.007843           # input scale factor

        blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight),
                (meanChannelVal, meanChannelVal, meanChannelVal), swapRBchannels, crop)

        # set this transformed image -> tensor blob as the network input

        net.setInput(blob)

        # perform forward inference on the network

        detections = net.forward()

        # process the detections from the CNN to give bounding boxes
        # i.e. for each detection returned from the network

        for i in range(detections.shape[2]):

            # extract the confidence of the detection

            confidence = detections[0, 0, i, 2]

            # provided that is above a threshold

            if confidence > confidence_threshold:

                # get the class number id and the bounding box

                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                # draw the bounding box on the frame

                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))

                # look up the class name based on the class id and draw it on the frame also

                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # display image

        cv2.imshow(windowName,frame)

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

        # e.g. if user presses "x" then exit / press "f" for fullscreen

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################
