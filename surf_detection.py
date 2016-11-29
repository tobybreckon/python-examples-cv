#####################################################################

# Example : SURF / SIFT detection from a video file specified on the
# command line (e.g. python FILE.py video_file) or from an
# attached web camera (default to SIFT)

# N.B. use mouse to select region, press h for homography calculation
# press s to switch from SURF to SIFT

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2016 Toby Breckon
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# based in part on tutorial at:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

# version 0.2 - fixed bug with segmentation faults on FLANN object rebuilds

#####################################################################

import cv2
import sys
import math
import numpy as np

#####################################################################

keep_processing = True;
camera_to_use = 1; # 0 if you have one camera, 1 or > 1 otherwise

selection_in_progress = False; # support interactive region selection

compute_object_position_via_homography = False;  # compute homography ?
MIN_MATCH_COUNT = 10; # number of matches to compute homography

#####################################################################

# select a region using the mouse

boxes = [];
current_mouse_position = np.ones(2, dtype=np.int32);

def on_mouse(event, x, y, flags, params):

    global boxes;
    global selection_in_progress;

    current_mouse_position[0] = x;
    current_mouse_position[1] = y;

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = [];
        # print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [x, y];
        selection_in_progress = True;
        boxes.append(sbox);

    elif event == cv2.EVENT_LBUTTONUP:
        # print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y];
        selection_in_progress = False;
        boxes.append(ebox);

#####################################################################

# define video capture object

cap = cv2.VideoCapture();

# define display window name

windowName = "Live Camera Input"; # window name
windowName2 = "Feature Matches"; # window name
windowNameSelection = "Selected Features";

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((len(sys.argv) == 2) and (cap.open(str(sys.argv[1]))))
    or (cap.open(camera_to_use))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL);

    # set a mouse callback

    cv2.setMouseCallback(windowName, on_mouse, 0);
    cropped = False;

    # Create a SURF feature object with a Hessian Threshold set to 400
    # if this fails see (http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/)

    feature_object = cv2.xfeatures2d.SURF_create(400);

    # create a Fast Linear Approx. Nearest Neightbours (Kd-tree) object for
    # fast feature matching

    # FLANN_INDEX_KDTREE = 0;
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1); # for trees > 1, things break
    # search_params = dict(checks=50);   # or pass empty dictionary
    # matcher = cv2.FlannBasedMatcher(index_params,search_params);

    # ^ ^ ^ ^ yes - in an ideal world, but in the world where this issue
    # still remains open in OpenCV (https://github.com/opencv/opencv/issues/5667)
    # just use the slower Brute Force matcher and go to bed
    # summary: python OpenCV bindings issue, ok to use in C++

    matcher = cv2.BFMatcher();

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read();
            if frame is None:
                keep_processing = False;
                print("no frame available");
                quit();

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # convert to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

        #####################################################################

        # select region using the mouse and display it

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):

            # obtain cropped region as an image

            crop = frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()
            h, w, c = crop.shape;   # size of cropped region

            # if region is valid

            if (h > 0) and (w > 0):
                cropped = True;

                # detect features and compute associated descriptor vectors

                keypoints_cropped_region, descriptors_cropped_region = feature_object.detectAndCompute(crop,None);

                # display keypoints on the image

                cropped_region_with_features = cv2.drawKeypoints(crop,keypoints_cropped_region,None,(255,0,0),4);

                # display features on cropped region

                cv2.imshow(windowNameSelection,cropped_region_with_features);

            # reset list of boxes so we do this part only once

            boxes = [];

        # interactive display of selection box

        if (selection_in_progress):
            top_left = (boxes[0][0], boxes[0][1]);
            bottom_right = (current_mouse_position[0], current_mouse_position[1]);
            cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2);

        #####################################################################

        # if we have a selected region

        if (cropped):

           # detect and match features from current image

           keypoints, descriptors = feature_object.detectAndCompute(gray_frame,None);

           #print(len(descriptors));
           #print(len(descriptors_cropped_region));

           # get best matches (and second best matches) between current and cropped region features
           # using a k Nearst Neighboour (kNN) radial matcher with k=2

           matches = [];
           if (len(descriptors) > 0):
                #flann.clear();
                matches = matcher.knnMatch(descriptors_cropped_region, trainDescriptors = descriptors, k = 2)

           # Need to isolate only good matches, so create a mask

           matchesMask = [[0,0] for i in range(len(matches))]

           # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
           # using the matching distances of the first and second matches

           good_matches = [];
           for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0];
                    good_matches.append(m);

           # if set to compute homography also

           if (compute_object_position_via_homography):

            # check we have enough good matches

            if len(good_matches)>MIN_MATCH_COUNT:

                # construct two sets of points - source (the selected object/region points), destination (the current frame points)

                source_pts = np.float32([ keypoints_cropped_region[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2);
                destination_pts = np.float32([ keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2);

                # compute the homography (matrix transform) from one set to the other using RANSAC

                H, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)

                # extract the bounding co-ordinates of the cropped/selected region

                h,w,c = crop.shape;
                boundingbox_points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2);

                # transform the bounding co-ordinates by homography H

                dst = cv2.perspectiveTransform(boundingbox_points,H);

                # draw the corresponding

                frame = cv2.polylines(frame,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

            else:
                print("Not enough matches for found for homography - %d/%d" % (len(good_matches),MIN_MATCH_COUNT));

           # draw the matches

           draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0);
           display_matches = cv2.drawMatchesKnn(crop,keypoints_cropped_region,frame,keypoints,matches,None,**draw_params);
           cv2.imshow(windowName2,display_matches);

        #####################################################################

        # display live image

        cv2.imshow(windowName,frame);

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
        # here we use a wait time in ms. that takes account of processing time already used in the loop

        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF;

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False;

        # compute homography position for object

        elif (key == ord('h')):
            compute_object_position_via_homography = not(compute_object_position_via_homography);

        # use SIFT points

        elif (key == ord('s')):
            # Create a SIFT feature object with a Hessian Threshold set to 400
            # if this fails see (http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/)

            feature_object = cv2.xfeatures2d.SIFT_create(400);
            cropped = False;


    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");

#####################################################################
