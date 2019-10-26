###########################################################################################

# Example : perform EigenFace based face recognition using haar cascade detection
# for initial face localization within the image

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# recognition part based on earlier C example at:
# https://github.com/tobybreckon/c-examples-ipcv/blob/master/eigenimage_based_recognition.cc

# image loading part based on example at:
# https://www.learnopencv.com/eigenface-using-opencv-c-python/

# get trained cascade files from:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

# original academic references
# - face detection part [IJCV - Viola / Jones, 2004]
# - face recognition part [Pentland / Turk, 1991]

################################################################################

import cv2
import argparse
import sys
import os
import numpy as np
import math

################################################################################

keep_processing = True

################################################################################

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-e", "--eigenfaces", type=int, help="specify number of eigenface (PCA) dimensions to use", default=10)
parser.add_argument("-f", "--path_to_faces", type=str, help="path to face images", default='/tmp/images/')
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument("-p", "--portrait_percentage", type=int, help="for potrait style inputs, specify upper percentage of image in which to detect face", default=100)
parser.add_argument("-s", "--face_size", type=int, help="specify height/width of face images to use for the input to the PCA", default=300)
parser.add_argument("-es", "--eigenfaces_to_skip", type=int, help="skip the first N eigenface dimensions that normally contain illumination information only", default=3)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

################################################################################
# Read images from the directory

def readImages(path, haar_face_detector):
    print("Reading images from " + path, end="...")
    cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
    # Create array of array of images and names
    images = []
    names = []
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        name = os.path.splitext(filePath)[0]
        if fileExt in [".jpg", ".jpeg", ".png"]:

            # load image

            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)
            if im is None :
                print("image:{} not read properly".format(imagePath))
                continue

            # assume 1 face per image, detect using haar, find in top N% of image

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            face = haar_face_detector.detectMultiScale(gray[0:int(height * (args.portrait_percentage/100)), 0:width],
                            scaleFactor=1.1, minNeighbors=4, minSize=(60,60), flags=cv2.CASCADE_DO_CANNY_PRUNING)

            if (len(face) > 0):
                (x,y,w,h) = face[0]
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (args.face_size, args.face_size))

                # try to compensate for illumination variance using histogram equalization

                roi_gray = cv2.equalizeHist(roi_gray)

                # Add image to list
                # (once only here, but could also add flips or other transforms to make it more robust)

                images.append(roi_gray)
                names.append(name)

                cv2.imshow("face", roi_gray)
                cv2.waitKey(100)

            else:
                print("image:{} - no face detected.".format(imagePath))

    cv2.destroyWindow("face")

    if len(images) == 0 :
        print("No facws found in image set: " + path)
        sys.exit(0)

    print(str(len(images)) + " files read.")
    return (images, names)

################################################################################
# perform PCA on a set of images

def performPCA(images):

    #  Allocate space for all images in one data matrix. The size of the data matrix is
    # ( w  * h  * c, numImages ) where, w = width of an image in the dataset.
    # h = height of an image in the dataset. c is for the number of color channels.

    numImages = len(images)
    sz = images[0].shape
    channels = 1 # grayescale
    data = np.zeros((numImages, sz[0] * sz[1] * channels), dtype=np.float32)

    # store images as floating point vectors normalized 0 -> 1

    for i in range(0, numImages):
        image = np.float32(images[i])/255.0
        data[i,:] = image.flatten() # N.B. data is stored as rows

    # compute the eigenvectors from the stack of image vectors created

    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=args.eigenfaces)

    # use the eigenvectors to project the set of images to the new PCA space representation

    coefficients = cv2.PCAProject(data, mean, eigenVectors)

    # calculate the covariance and mean of the PCA space representation of the images
    # (skipping the first N eigenfaces that often contain just illumination variance, default N=3 )

    covariance_coeffs, mean_coeffs = cv2.calcCovarMatrix(coefficients[:,args.eigenfaces_to_skip:args.eigenfaces], mean=None, flags=cv2.COVAR_NORMAL | cv2.COVAR_ROWS, ctype = cv2.CV_32F)

    return (mean, eigenVectors, coefficients, mean_coeffs, covariance_coeffs)

################################################################################
# return index of best matching face from set of all PCA projcted coefficients
# based on miniumum Mahalanobis (M) distance and this minimum M distance

def find_matching_face(face_coefficients_to_match, coefficients_of_all_faces, covariance):

    # set up loop variables

    nearest_face_index = 0
    nearest_face_distance = 100 # i.e. huge
    current_face = 0

    for pca_face_coefficient in coefficients_of_all_faces:

        # calculate the Mahalanobis distamce between the coefficients we need to match and each from the set of faces
        # (skipping the first N eigenfaces that often contain just illumination variance, default N=3 )

        m_dist = cv2.Mahalanobis(face_coefficients_to_match[:,args.eigenfaces_to_skip:args.eigenfaces], pca_face_coefficient.reshape(1,args.eigenfaces)[:,args.eigenfaces_to_skip:args.eigenfaces], np.linalg.inv(covariance))

        # alternatively use the L1 or L2 norm as per original [Pentland / Turk 1991] paper - which used L1
        # m_dist = numpy.linalg.norm(face_coefficients_to_match[:,3:args.eigenfaces]-pca_face_coefficient.reshape(1,args.eigenfaces)[:,3:args.eigenfaces])

        if (m_dist < nearest_face_distance):
            nearest_face_index = current_face
            nearest_face_distance = m_dist

        current_face += 1

    return (nearest_face_index, nearest_face_distance)

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

windowName = "Face Recognition using EigenFaces" # window name

# define haar cascade objects

# required cascade classifier files (and many others) available from:
# https://github.com/opencv/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if (face_cascade.empty()):
    print("Failed to load cascade from file.")
    sys.exit(0)

# load set of face images

(images, names) = readImages(args.path_to_faces, face_cascade)

# perform PCA on the images

(mean, eigenVectors, coefficients, mean_coeffs, covariance_coeffs) = performPCA(images)

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces using haar cascade trained on faces

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60), flags=cv2.CASCADE_DO_CANNY_PRUNING)

        # for each detected face, try to detect eyes inside the top
        # half of the face region face region

        for (x,y,w,h) in faces:

            # draw each face bounding box and extract regions of interest (roi)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # project detected face to PCA space

            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (args.face_size, args.face_size))
            roi_gray = cv2.equalizeHist(roi_gray)   # try to compensate for illumination variance
            roi_gray = np.float32(roi_gray)/255.0   # normalise as 0 -> 1

            face_coefficients = cv2.PCAProject(roi_gray.flatten().reshape(1, args.face_size * args.face_size), mean, eigenVectors)

            # measure distance to PCA coefficient for each face and find best match

            face_index, face_distance = find_matching_face(face_coefficients, coefficients, covariance_coeffs)

            # show best match / display name and Mahalanobis distance for best match

            cv2.putText(frame, names[face_index] + ": " + str(round(face_distance, 2)), (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # display stored equalizeHist version side-by-side

            cv2.imshow("best match", images[face_index])

        # display image

        cv2.imshow(windowName,frame)
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
