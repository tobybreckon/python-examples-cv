#!/bin/sh

################################################################################

# run a batch test over all the examples from the bash shell (linux)

# Copyright (c) 2019 Dept Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

################################################################################

PYTHON_INTERPRETATOR=python3
CAM_TO_TEST=0
VIDEO_TO_TEST=video.avi

echo
echo Using $PYTHON_INTERPRETATOR with camera $CAM_TO_TEST and video $VIDEO_TO_TEST
echo "Running test suite - press 'x' in OpenCV window to exist each example."
echo

# get testing resouces if they do not exist

[ -f example.jpg ] || { wget https://upload.wikimedia.org/wikipedia/commons/b/b4/JPEG_example_JPG_RIP_100.jpg; mv JPEG_example_JPG_RIP_100.jpg example.jpg; }
[ -f video.avi ] || { wget http://clips.vorwaerts-gmbh.de/big_buck_bunny.mp4; mv big_buck_bunny.mp4 video.avi; }

################################################################################

# run defaults

echo "Running default tests ..."
echo

for example in *.py
do
 echo "Testing example: " $example
 $PYTHON_INTERPRETATOR $example
 echo
done

################################################################################

# run cam test

echo "Running camera based tests ..."
echo

for example in *.py
do
 echo "Testing example: " $example -c $CAM_TO_TEST
 $PYTHON_INTERPRETATOR $example -c $CAM_TO_TEST
 echo
done

################################################################################

# run cam test and resize

echo "Running camera based tests with resizing ..."
echo

for example in *.py
do
 echo "Testing example: " $example -c $CAM_TO_TEST -r 0.25
 $PYTHON_INTERPRETATOR $example -c $CAM_TO_TEST -r 0.25
 echo
done


################################################################################

# run video file test

echo "Running video file based tests ..."
echo

for example in *.py
do
 echo "Testing example: " $example $VIDEO_TO_TEST
 $PYTHON_INTERPRETATOR $example $VIDEO_TO_TEST
 echo
done

################################################################################
