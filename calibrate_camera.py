#####################################################################

# Example : perform intrinsic calibration of a  connected camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018-2021 Department of Computer Science,
#                         Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements:

# http://opencv-python-tutroals.readthedocs.org/en/latest/ \
# py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

# http://docs.ros.org/electric/api/cob_camera_calibration/html/calibrator_8py_source.html

#####################################################################

import cv2
import argparse
import sys
import numpy as np

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
    "-cbx",
    "--chessboardx",
    type=int,
    help="specify number of internal chessboard squares \
            (corners) in x-direction",
    default=6)
parser.add_argument(
    "-cby",
    "--chessboardy",
    type=int,
    help="specify number of internal chessboard squares \
            (corners) in y-direction",
    default=8)
parser.add_argument(
    "-cbw",
    "--chessboardw",
    type=float,
    help="specify width/height of chessboard squares in mm",
    default=40.0)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    help="specify number of iterations for each stage of optimisation",
    default=100)
parser.add_argument(
    "-e",
    "--minimum_error",
    type=float,
    help="specify lower error threshold upon which to stop \
        optimisation stages",
    default=0.001)
args = parser.parse_args()

#####################################################################

#  define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    import camera_stream
    cap = camera_stream.CameraVideoStream()

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window names

window_name = "Camera Input"  # window name
window_nameU = "Undistored (calibrated) Camera"  # window name

#####################################################################

# perform intrinsic calibration (removal of image distortion in image)

do_calibration = False
termination_criteria_subpix = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    args.iterations,
    args.minimum_error)

# set up a set of real-world "object points" for the chessboard pattern

patternX = args.chessboardx
patternY = args.chessboardy
square_size_in_mm = args.chessboardw

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX * patternY, 3), np.float32)
objp[:, :2] = np.mgrid[0:patternX, 0:patternY].T.reshape(-1, 2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

#####################################################################

# count number of chessboard detections
chessboard_pattern_detections = 0

print()
print("--> hold up chessboard (grabbing images at 2 fps)")
print("press c : to continue to calibration")

#####################################################################

# open connected camera

if cap.open(args.camera_to_use):

    # override default camera resolution

    if (args.set_resolution is not None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.set_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.set_resolution[0])

    print("INFO: input resolution : (",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), ")")

    while (not (do_calibration)):

        # grab frames from camera

        ret, frame = cap.read()

        # rescale if specified

        if (args.rescale != 1.0):
            frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # convert to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners in the image
        # (change flags to perhaps improve detection ?)

        ret, corners = cv2.findChessboardCorners(
            gray, (patternX, patternY), None, cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)

        if (ret):

            chessboard_pattern_detections += 1

            # add object points to global list

            objpoints.append(objp)

            # refine corner locations to sub-pixel accuracy and then

            corners_sp = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), termination_criteria_subpix)
            imgpoints.append(corners_sp)

            # Draw and display the corners

            drawboard = cv2.drawChessboardCorners(
                frame, (patternX, patternY), corners_sp, ret)

            text = 'detected: ' + str(chessboard_pattern_detections)
            cv2.putText(drawboard, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

            cv2.imshow(window_name, drawboard)
        else:
            text = 'detected: ' + str(chessboard_pattern_detections)
            cv2.putText(frame, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

            cv2.imshow(window_name, frame)

        # start the event loop

        key = cv2.waitKey(500) & 0xFF  # wait 500 ms. between frames
        if (key == ord('c')):
            do_calibration = True

else:
    print("Cannot open connected camera.")
    exit()

#####################################################################

# check we detected some patterns within the first loop

if (chessboard_pattern_detections == 0):
    print("No calibration patterns detected - exiting.")
    exit()

#####################################################################

# perform calibration - uses [Zhang, 2000]

print("START - intrinsic calibration ...")

ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("FINISHED - intrinsic calibration")

# print output in readable format

print()
print("Intrinsic Camera Calibration Matrix, K - from intrinsic calibration:")
print("(format as follows: fx, fy - focal lengths / cx, cy - optical centers)")
print("[fx, 0, cx]\n[0, fy, cy]\n[0,  0,  1]")
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(K)
print()
print("Intrinsic Distortion Co-effients, D - from intrinsic calibration:")
print("(k1, k2, k3 - radial p1, p2 - tangential - distortion coefficients)")
print("[k1, k2, p1, p2, k3]")
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
print(D)
print()
print("Image resolution used (width, height): ", np.flip(frame.shape[:2]))

#####################################################################

# perform undistortion (i.e. calibration) of the images

keep_processing = True

print()
print("-> performing undistortion")
print("press x : to exit")

while (keep_processing):

    # grab frames from camera

    ret, frame = cap.read()

    # undistort image using camera matrix K and distortion coefficients D

    undistorted = cv2.undistort(frame, K, D, None, None)

    # display both images

    cv2.imshow(window_name, frame)
    cv2.imshow(window_nameU, undistorted)

    # start the event loop - essential

    key = cv2.waitKey(40) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if (key == ord('x')):
        keep_processing = False

#####################################################################

# close all windows and cams.

cv2.destroyAllWindows()

#####################################################################
