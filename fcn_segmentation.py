##########################################################################

# Example : perform FCN semantic image segmentation from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera (FCN segmentation: Long et al, CVPR 2015)

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# This code: significant portions based on the example available at:
# https://github.com/opencv/opencv/blob/master/samples/dnn/segmentation.py


# Copyright (c) 2021 Toby Breckon, Dept. Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

##########################################################################

# To use download the following files:

# http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
# https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/fcn8s-heavy-pascal.prototxt
# https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_pascal_voc.txt

##########################################################################

import cv2
import argparse
import sys
import math
import numpy as np

##########################################################################

keep_processing = True
colors = None

##########################################################################

# generate and display colour legend for segmentation classes


def generate_legend(classes, height):
    blockHeight = math.floor(height/len(classes))

    legend = np.zeros((blockHeight * len(colors), 200, 3), np.uint8)
    for i in range(len(classes)):
        block = legend[i * blockHeight:(i + 1) * blockHeight]
        block[:, :] = colors[i]
        cv2.putText(block, classes[i],
                    (0, blockHeight//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return legend

##########################################################################

# concatenate two RGB/grayscale images horizontally (left to right)
# handling differing channel numbers or image heights in the input


def h_concatenate(img1, img2):

    # get size and channels for both images

    height1 = img1.shape[0]

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

    # height of first image is master height, width can remain unchanged

    if (height1 != height2):
        out2 = cv2.resize(out2, (width2, height1))

    return np.hstack((out1, out2))


##########################################################################

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
    default='cpu')
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

    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

window_name = "FCN Semantic Image Segmentation"  # window name

##########################################################################

# Load names of class labels (background = class 0, for PASCAL VOC)

classes = None
with open("object_detection_classes_pascal_voc.txt", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
classes.insert(0, "background")  # insery a background class as 0

##########################################################################

# Load CNN model

net = cv2.dnn.readNet(
    "fcn8s-heavy-pascal.caffemodel",
    "fcn8s-heavy-pascal.prototxt",
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

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        #######################################################################
        # FCN Segmentation:
        # model: "fcn8s-heavy-pascal.caffemodel"
        # config: "fcn8s-heavy-pascal.prototxt"
        # mean: [0, 0, 0]
        # scale: 1.0
        # width: 500
        # height: 500
        # rgb: false
        # object_detection_classes_pascal_voc.txt
        # classes:
        #######################################################################

        # create a 4D tensor "blob" from a frame.

        blob = cv2.dnn.blobFromImage(
                                     frame, scalefactor=1.0,
                                     size=(500, 500), mean=[0, 0, 0],
                                     swapRB=False, crop=False
                                    )

        # Run forward inference on the model

        net.setInput(blob)
        result = net.forward()

        numClasses = result.shape[1]
        height = result.shape[2]
        width = result.shape[3]

        # define colours

        if not colors:
            np.random.seed(888)
            colors = [np.array([0, 0, 0], np.uint8)]
            for i in range(1, numClasses + 1):
                colors.append((colors[i - 1] +
                              np.random.randint(0, 256, [3],
                              np.uint8)) / 2
                              )
            del colors[0]

            # generate legend
            legend = generate_legend(classes, frameHeight)

        # display segmentation

        classIds = np.argmax(result[0], axis=0)
        segm = np.stack([colors[idx] for idx in classIds.flatten()])
        segm = segm.reshape(height, width, 3)

        segm = cv2.resize(segm, (frameWidth, frameHeight),
                          interpolation=cv2.INTER_NEAREST)

        # stop the timer and convert to ms. (to see how long processing and
        # display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # Display efficiency information

        label = ('Inference time: %.2f ms' % stop_t) + \
            (' (Framerate: %.2f fps' % (1000 / stop_t)) + ')'
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image(s) as concatenated single image

        cv2.imshow(window_name,
                   h_concatenate(h_concatenate(frame, segm.astype(np.uint8)),
                                 legend))
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & args.fullscreen)

        # start the event loop - essential

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
