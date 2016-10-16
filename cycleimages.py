#####################################################################

# Example : load and display a set of images from a directory
# basic illustrative python script

# For use with provided test / training datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# version: 0.2 (bug fix on waitKey() usage - line 39)
# version: 0.3 (python 3 - line 33)

#####################################################################

import cv2
import os

directory_to_cycle = "directory-name-goes-here" # edit this

#####################################################################

# display all images in directory

for filename in os.listdir(directory_to_cycle):

    # if it is a PNG file

    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename));

        # read it and display in a window

        img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
        cv2.imshow('the image',img)
        key = cv2.waitKey(200) # wait 200ms
        if (key == ord('x')):
            break


# close all windows

cv2.destroyAllWindows()

#####################################################################
