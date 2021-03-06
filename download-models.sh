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
          https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
          https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
          https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml
        )

# associated MD5 checksums (output of md5sum filename)

MD5SUMS=( "4fdfb6d202e9d8e65da14c78b604af95  classification_classes_ILSVRC2012.txt"
          "8fc50561361f8bcf96b0177086e7616c  coco.names"
          "81d7d9cb3438456214afcdb5c83e7bfb  object_detection_classes_coco.txt"
          "c9e6e28e5b84b7b49c436f929b58db91  pose_deploy_linevec.prototxt"
          "5156d31f670511fce9b4e28b403f2939  pose_iter_440000.caffemodel"
          "0357e4e11d173c72a01615888826bc8e  squeezenet_v1.1.caffemodel"
          "dfe9c8d69b154f0ebbba87bc32371e2d  squeezenet_v1.1.prototxt"
          "5d442b0e550e6c640068e7e15e498599  yolov3.cfg"
          "c84e5b99d0e52cd466ae710cadf6d84c  yolov3.weights"
          "1f1902262c16c2d9acb9bc4f8a8c266f  faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
          "2d6fac0caaec1f9558872755ff34818d  haarcascade_eye.xml"
          "a03f92a797e309e76e6a034ab9e02616  haarcascade_frontalface_default.xml"
          "acee557d79a3684cac72ebd811a4eee0  lbpcascade_frontalface_improved.xml"
        )

################################################################################

DIR_LOCAL_TARGET=/tmp/python-examples-cv-models
PWD_SCRIPT=`pwd`

################################################################################

# Preset this script to fail on error

set -e

# check for required commands to download and md5 check

(command -v curl | grep curl > /dev/null) ||
  (echo "Error: curl command not found, cannot download.")

  (command -v md5sum | grep md5sum > /dev/null) ||
    (echo "Error: md5sum command not found, cannot verify files.")


################################################################################

# Download - perform download of each model

mkdir -p $DIR_LOCAL_TARGET
cd $DIR_LOCAL_TARGET

for URL in ${MODELS[@]}; do
  echo
  echo "Downloading ... " $URL " -> " $DIR_LOCAL_TARGET/
  curl -L -k -O --remote-name $URL
done

# un-tar/gz any models that need this

for GZT in `ls *tar.gz`; do
  tar -xzf $GZT
  rm $GZT
done

cd $PWD_SCRIPT

################################################################################

# Post Download - check md5sum

cd $DIR_LOCAL_TARGET
echo
echo "Performing MD5 file verification checks ..."
printf '%s\n' "${MD5SUMS[@]}" > md5sums.txt
md5sum -c md5sums.txt

# Post Download - link all files to current directory

cd $PWD_SCRIPT
echo
echo "Linking files to current directory ..."
ln -sv $DIR_LOCAL_TARGET/* .

################################################################################
