################################################################################

# multi model file downloader - (c) 2021 Toby Breckon, Durham University, UK

################################################################################

# models and associated files for automated download

MODELS=(  https://pjreddie.com/media/files/yolov3.weights
          https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
          https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
          https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt
          https://github.com/forresti/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
          https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/squeezenet_v1.1.prototxt
          https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_coco.txt
          https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt
          http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
          http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
          https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
        )

################################################################################

DIR_LOCAL_TARGET=/tmp/python-examples-cv-models
PWD_SCRIPT=`pwd`

################################################################################

# Preset this script to fail on error

set -e

# check for required commands to download and md5 check

(command -v curl | grep curl > /dev/null) ||
  (echo "Error: curl command not found, cannot download!")

################################################################################

# Download - perform download of each model

mkdir -p $DIR_LOCAL_TARGET
cd $DIR_LOCAL_TARGET

for URL in ${MODELS[@]}; do
  echo
  echo "Downloading ... " $URL " -> " $DIR_LOCAL_TARGET/
  curl -L -k -O --remote-name $URL
done

# un-tar/gz any models needed so

for GZT in `ls *tar.gz`; do
  tar -xzf $GZT
  rm $GZT
done

cd $PWD_SCRIPT

################################################################################

# Post Download - link all files to current directory

echo
echo "Linking files to current directory ..."
ln -sv $DIR_LOCAL_TARGET/* .

################################################################################
