#####################################################################

# Example : stereo vision from 2 connected cameras using Semi-Global
# Block Matching

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements:

# http://opencv-python-tutroals.readthedocs.org/en/latest/ \
# py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

# http://docs.ros.org/electric/api/cob_camera_calibration/html/calibrator_8py_source.html

# OpenCV 3.0 example - stereo_match.py

# version 0.2 - fixed disparity scaling issues (March 2016)

#####################################################################

# TODO:

# add slider for some parameters
# add StereoBM option
# add load/save parameters using NumPy:
# http://docs.scipy.org/doc/numpy/reference/routines.io.html

#####################################################################

import cv2
import sys
import numpy as np

#####################################################################

keep_processing = True;
# camera_to_use = 0; # 0 if you have no built in webcam, 1 otherwise
camera_to_use = cv2.CAP_XIAPI; # for the Xiema cameras (opencv built with driver)

#####################################################################

# STAGE 1 - open 2 connected cameras

# define video capture object

camL = cv2.VideoCapture();
camR = cv2.VideoCapture();

# define display window names

windowNameL = "LEFT Camera Input"; # window name
windowNameR = "RIGHT Camera Input"; # window name

print("s : swap cameras left and right")
print("c : continue to next stage")

if ((camL.open(camera_to_use)) and (camR.open(camera_to_use + 1))):

    while (keep_processing):

        # grab frames from camera (to ensure best time sync.)

        camL.grab();
        camR.grab();

        # then retrieve the images in slow(er) time
        # (do not be tempted to use read() !)

        ret, frameL = camL.retrieve();
        ret, frameR = camR.retrieve();

        # display image

        cv2.imshow(windowNameL,frameL);
        cv2.imshow(windowNameR,frameR);

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('c')):
            keep_processing = False;
        elif (key == ord('s')):
            # swap the cameras if specified
            tmp = camL;
            camL = camR;
            camR = tmp;

else:
    print("Cannot open pair of cameras connected.")

#####################################################################

# STAGE 2: perform intrinsic calibration (removal of image distortion in each image)

do_calibration = False;
termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# set up a set of real-world "object points" for the chessboard pattern

patternX = 6;
patternY = 9;
square_size_in_mm = 40;

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX*patternY,3), np.float32)
objp[:,:2] = np.mgrid[0:patternX,0:patternY].T.reshape(-1,2)
objp = objp * square_size_in_mm;

# create arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsR = [] # 2d points in image plane.
imgpointsL = [] # 2d points in image plane.

# count number of chessboard detection (across both images)
chessboard_pattern_detections = 0;

print()
print("--> hold up chessboard")

while (not(do_calibration)):

        # grab frames from camera (to ensure best time sync.)

        camL.grab();
        camR.grab();

        # then retrieve the images in slow(er) time
        # (do not be tempted to use read() !)

        retR, frameL = camL.retrieve();
        retL, frameR = camR.retrieve();

        # convert to grayscale

        grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY);

        # Find the chess board corners in the image
        # (change flags to perhaps improve detection ?)

        retR, cornersL = cv2.findChessboardCorners(grayL, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE);
        retL, cornersR = cv2.findChessboardCorners(grayR, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE);

        # If found, add object points, image points (after refining them)

        if ((retR == True) and (retL == True)):

            chessboard_pattern_detections += 1;

            # add object points to global list

            objpoints.append(objp);

            # refine corner locations to sub-pixel accuracy and then

            corners_sp_L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),termination_criteria_subpix);
            imgpointsL.append(corners_sp_L);
            corners_sp_R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),termination_criteria_subpix);
            imgpointsR.append(corners_sp_R);

            # Draw and display the corners

            drawboardL = cv2.drawChessboardCorners(frameL, (patternX,patternY), corners_sp_L,retL)
            drawboardR = cv2.drawChessboardCorners(frameR, (patternX,patternY), corners_sp_R,retR)

            text = 'detected: ' + str(chessboard_pattern_detections);
            cv2.putText(drawboardL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8);

            cv2.imshow(windowNameL,drawboardL);
            cv2.imshow(windowNameR,drawboardR);
        else:
            text = 'detected: ' + str(chessboard_pattern_detections);
            cv2.putText(frameL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8);

            cv2.imshow(windowNameL,frameL);
            cv2.imshow(windowNameR,frameR);

        # start the event loop

        key = cv2.waitKey(100) & 0xFF; # wait 500ms between frames
        if (key == ord('c')):
            do_calibration = True;

# perform calibration on both cameras - uses [Zhang, 2000]

print("START - intrinsic calibration ...")

ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None);
ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None);

print("FINSIHED - intrinsic calibration")

# perform undistortion of the images

keep_processing = True;

print()
print("-> performing undistortion")

while (keep_processing):

    # grab frames from camera (to ensure best time sync.)

    camL.grab();
    camR.grab();

    # then retrieve the images in slow(er) time
    # (do not be tempted to use read() !)

    ret, frameL = camL.retrieve();
    ret, frameR = camR.retrieve();

    undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
    undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

    # display image

    cv2.imshow(windowNameL,undistortedL);
    cv2.imshow(windowNameR,undistortedR);

    # start the event loop - essential

    key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if (key == ord('c')):
        keep_processing = False;

# show mean re-projection error of the object points onto the image(s)

tot_errorL = 0
for i in range(len(objpoints)):
    imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    errorL = cv2.norm(imgpointsL[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
    tot_errorL += errorL

print("LEFT: Re-projection error: ", tot_errorL/len(objpoints))

tot_errorR = 0
for i in range(len(objpoints)):
    imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    errorR = cv2.norm(imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
    tot_errorR += errorR

print("RIGHT: Re-projection error: ", tot_errorR/len(objpoints))

#####################################################################

# STAGE 3: perform extrinsic calibration (recovery of relative camera positions)

# this takes the existing calibration parameters used to undistort the individual images as
# well as calculated the relative camera positions - represented via the fundamental matrix, F

# alter termination criteria to (perhaps) improve solution - ?

termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

print()
print("START - extrinsic calibration ...")
(rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F) = \
cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,  grayL.shape[::-1], criteria=termination_criteria_extrinsics, flags=0);

print("START - extrinsic calibration ...")

print("STEREO: RMS left to  right re-projection error: ", rms_stereo)

#####################################################################

# STAGE 4: rectification of images (make scan lines align left <-> right

# N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
# only valid pixels are visible (no black areas after rectification). alpha=1 means
# that the rectified image is decimated and shifted so that all the pixels from the original images
# from the cameras are retained in the rectified images (no source image pixels are lost)." - ?

RL, RR, PL, PR, _, _, _ = cv2.stereoRectify(camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,  grayL.shape[::-1], R, T, alpha=-1);

# compute the pixel mappings to the rectified versions of the images

mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_matrix_l, dist_coeffs_l, RL, PL, grayL.shape[::-1], cv2.CV_32FC1);
mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, RR, PR, grayL.shape[::-1], cv2.CV_32FC1);

print()
print("-> performing rectification")

keep_processing = True;

while (keep_processing):

    # grab frames from camera (to ensure best time sync.)

    camL.grab();
    camR.grab();

    # then retrieve the images in slow(er) time
    # (do not be tempted to use read() !)

    ret, frameL = camL.retrieve();
    ret, frameR = camR.retrieve();

    # undistort and rectify based on the mappings (could improve interpolation and image border settings here)

    undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR);
    undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR);

    # display image

    cv2.imshow(windowNameL,undistorted_rectifiedL);
    cv2.imshow(windowNameR,undistorted_rectifiedR);

    # start the event loop - essential

    key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit

    if (key == ord('c')):
        keep_processing = False;

#####################################################################

# STAGE 5: calculate stereo depth information

# uses a modified H. Hirschmuller algorithm [HH08] that differs (see opencv manual)

# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

print()
print("-> calc. disparity image")

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

keep_processing = True;

windowNameD = "SGBM Stereo Disparity - Output"; # window name

while (keep_processing):

    # grab frames from camera (to ensure best time sync.)

    camL.grab();
    camR.grab();

    # then retrieve the images in slow(er) time
    # (do not be tempted to use read() !)

    ret, frameL = camL.retrieve();
    ret, frameR = camR.retrieve();

    # remember to convert to grayscale (as the disparity matching works on grayscale)

    grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY);

    # undistort and rectify based on the mappings (could improve interpolation and image border settings here)
    # N.B. mapping works independant of number of image channels

    undistorted_rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR);
    undistorted_rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR);

    # compute disparity image from undistorted and rectified versions
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(undistorted_rectifiedL,undistorted_rectifiedR);
    cv2.filterSpeckles(disparity, 0, 4000, 128);

    # scale the disparity to 8-bit for viewing

    disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

    # display image

    cv2.imshow(windowNameL,undistorted_rectifiedL);
    cv2.imshow(windowNameR,undistorted_rectifiedR);

    #display disparity

    cv2.imshow(windowNameD, disparity_scaled);

    # start the event loop - essential

    key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit

    if (key == ord('c')):
        keep_processing = False;

#####################################################################

# close all windows and cams.

cv2.destroyAllWindows()

#####################################################################
