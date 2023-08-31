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
          https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel
          https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
          https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
          https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
          https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml
          http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
          https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
          http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
          https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/fcn8s-heavy-pascal.prototxt
          https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_pascal_voc.txt
          https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel
          https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt
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
          "5708e4e579d8e4eabeec6c555d4234b2  mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
          "b47e443b313a709e4c39c1caeaa3ecb3  mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
          "c03b2953ebd846c270da1a8e8f200c09  fcn8s-heavy-pascal.caffemodel"
          "532698b83c2e8fa5a010bd996d19d30a  fcn8s-heavy-pascal.prototxt"
          "5ae5d62183cfb6f6d3ac109359d06a1b  object_detection_classes_pascal_voc.txt"
          "8bed6fa43361685f4c78f1c084be7775  MobileNetSSD_deploy.caffemodel"
          "aa2a13fe1fba2c3b7e067067a6749e7e  MobileNetSSD_deploy.prototxt"

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
rm -f md5sums.txt

# Post Download - link all files to current directory

cd $PWD_SCRIPT
echo
echo "Linking files to current directory ..."
ln -sv $DIR_LOCAL_TARGET/* .

################################################################################
