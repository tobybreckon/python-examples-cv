##########################################################################

# threaded frame capture from camera to avoid camera frame buffering delays
# (always delivers the latest frame from the camera)

# Copyright (c) 2018-2021 Toby Breckon, Durham University, UK
# Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com
# MIT License (MIT)

# based on code from this tutorial, with changes to make object method call
# compatible with cv2.VideoCapture(src) as far as possible, optional OpenCV
# Transparent API support (disabled by default) and improved thread management:
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

##########################################################################

# suggested basic usage - as per example in canny.py found at:
# https://github.com/tobybreckon/python-examples-cv/blob/master/canny.py

# try:
#    import camera_stream
#    cap = camera_stream.CameraVideoStream()
#    print("INFO: using CameraVideoStream() threaded capture")
# except BaseException:
#    print("INFO: CameraVideoStream() module not found")
#    cap = cv2.VideoCapture()

# in the above example this makes use of the CameraVideoStream if it is
# available (i.e. camera_stream.py is in the module search path) and
# falls back to using cv2.VideoCapture otherwise

# use with other OpenCV video backends just explicitly pass the required
# OpenCV flag as follows:
#    ....
#    import camera_stream
#    cap = camera_stream.CameraVideoStream()
#    ....
#
#    cap.open("your | gstreamer | pipeline", cv2.CAP_GSTREAMER)
#
# Ref: https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html

# OpenCV T-API usage - alternative usage to enable OpenCV Transparent API
# h/w acceleration where available on all subsequent processing of image
# ....
#    import camera_stream
#    cap = camera_stream.CameraVideoStream(use_tapi=True)
# ....

##########################################################################

# import the necessary packages

from threading import Thread
import cv2
import sys
import atexit
import logging

##########################################################################

# handle older versions of OpenCV (that had a different constuctor
# prototype for cv2.VideoCapture() it appears) semi-gracefully

(majorCV, minorCV, _) = cv2.__version__.split(".")
if ((majorCV <= '3') and (minorCV <= '4')):
    raise NameError('OpenCV version < 3.4,'
                    + ' not compatible with CameraVideoStream()')

##########################################################################

# set up logging

log_level = logging.CRITICAL  # change to .INFO / .DEBUG for useful info

log_msg_format = '%(asctime)s - Thead ID: %(thread)d - %(message)s'
logging.basicConfig(format=log_msg_format, level=log_level)

##########################################################################

# set up global variables and atexit() function to facilitate safe thread exit
# without a segfault from the VideoCapture object as experienced on some
# platforms
# (as __del__ and __exit__ are not called outside a 'with' construct)

exitingNow = False  # global flag for program exit
threadList = []    # list of current threads (i.e. multi-camera/thread safe)

###########################


def closeDownAllThreadsCleanly():
    global exitingNow
    global threadList

# set exit flag to cause each thread to exit

    exitingNow = True

# for each thread wait for it to exit

    for thread in threadList:
        thread.join()

###########################


atexit.register(closeDownAllThreadsCleanly)

##########################################################################


class CameraVideoStream:
    def __init__(self, src=None, backend=None,
                 name="CameraVideoStream", use_tapi=False):

        # initialize the thread name
        self.name = name

        # initialize the variables used to indicate if the thread should
        # be stopped or suspended
        self.stopped = False
        self.suspend = False

        # set these to null values initially
        self.grabbed = 0
        self.frame = None
        self.threadID = -1

        # set the initial timestamps to zero
        self.timestamp = 0
        self.timestamp_last_read = 0

        # set internal framecounters to -1
        self.framecounter = -1
        self.framecounter_last_read = -1

        # set OpenCV Transparent API usage

        self.tapi = use_tapi

        # set some sensible backends for real-time video capture from
        # directly connected hardware on a per-OS basis,
        # that can we overidden via the open() method
        if sys.platform.startswith('linux'):        # all Linux
            self.backend_default = cv2.CAP_V4L
        elif sys.platform.startswith('win'):        # MS Windows
            self.backend_default = cv2.CAP_DSHOW
        elif sys.platform.startswith('darwin'):     # macOS
            self.backend_default = cv2.CAP_AVFOUNDATION
        else:
            self.backend_default = cv2.CAP_ANY      # auto-detect via OpenCV

        # if a source was specified at init, proceed to open device
        if not (src is None):
            self.open(src, backend)

    def open(self, src, backend=None):

        # determine backend to specified by user
        if (backend is None):
            backend = self.backend_default

        # check if aleady opened via init method
        if (self.grabbed > 0):
            return True

        # initialize the video camera stream
        self.camera = cv2.VideoCapture(src, backend)

        # when the backend is v4l (linux) set the buffer size to 1
        # (as this is implemented for this backend and not others)
        if (backend == cv2.CAP_V4L):
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # read the first frame from the stream (and its timestamp)
        (self.grabbed, self.frame) = self.camera.read()
        self.timestamp = self.camera.get(cv2.CAP_PROP_POS_MSEC)
        self.framecounter += 1
        logging.info("CAM %d - GRAB - frame %d @ time %f",
                      self.threadID, self.framecounter, self.timestamp)

        # only start the thread if in-fact the camera read was successful
        if (self.grabbed):
            # create the thread to read frames from the video stream
            thread = Thread(target=self.update, name=self.name, args=())

            #  append thread to global array of threads
            threadList.append(thread)

            # get thread id we will use to address thread on list
            self.threadID = len(threadList) - 1

            # start thread and set it to run in background
            threadList[self.threadID].daemon = True
            threadList[self.threadID].start()

        return (self.grabbed > 0)

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set or exiting, stop the
            # thread
            if self.stopped or exitingNow:
                self.grabbed = 0  # set flag to ensure isOpen() returns False
                self.camera.release()  # cleanly release camera hardware
                return

            # otherwise, read the next frame from the stream
            # provided we are not suspended (and get timestamp)

            if not (self.suspend):
                self.camera.grab()
                latest_timestamp = self.camera.get(cv2.CAP_PROP_POS_MSEC)
                if (latest_timestamp > self.timestamp):
                    (self.grabbed, self.frame) = self.camera.retrieve()
                    self.framecounter += 1
                    logging.info("CAM %d - GRAB - frame %d @ time %f",
                                 self.threadID, self.framecounter, latest_timestamp)
                    logging.debug("CAM %d - GRAB - inter-frame diff (ms) %f",
                                  self.threadID, latest_timestamp - self.timestamp)
                    self.timestamp = latest_timestamp
                else:
                    logging.info("CAM %d - GRAB - same timestamp skip %d",
                                 self.threadID, latest_timestamp)

    def grab(self):
        # return status of most recent grab by the thread
        return self.grabbed

    def retrieve(self):
        # same as read() in the context of threaded capture
        return self.read()

    def read(self):

        # remember the timestamp/count of the lastest image returned by read()
        # so that subsequent calls to .get() can return the timestamp
        # that is consistent with the last image the caller got via read()
        frame_offset = (self.framecounter - self.framecounter_last_read)
        self.timestamp_last_read = self.timestamp
        self.framecounter_last_read = self.framecounter

        for skip in range(1, frame_offset):
            logging.info("CAM %d - SKIP - frame %d", self.threadID, self.framecounter_last_read
                         - frame_offset + skip)

        logging.info("CAM %d - READ - frame %d @ time %f",
                      self.threadID, self.framecounter, self.timestamp)

        # return the frame most recently read
        if (self.tapi):
            # return OpenCV Transparent API UMat frame for H/W acceleration
            return (self.grabbed, cv2.UMat(self.frame))
        # return standard numpy frame
        return (self.grabbed, self.frame)

    def isOpened(self):
        # indicate that the camera is open successfully
        return (self.grabbed > 0)

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def set(self, property_name, property_value):
        # set a video capture property (behavior as per OpenCV manual for
        # VideoCapture)

        # first suspend thread
        self.suspend = True

        # set value - wrapping it in grabs() so it takes effect
        self.camera.grab()
        ret_val = self.camera.set(property_name, property_value)
        self.camera.grab()

        # whilst we are still suspended flush the frame buffer held inside
        # the object by reading a new frame with new settings otherwise a race
        # condition will exist between the thread's next call to update() after
        # it un-suspends and the next call to read() by the object user
        (self.grabbed, self.frame) = self.camera.read()
        self.timestamp = self.camera.get(cv2.CAP_PROP_POS_MSEC)
        self.framecounter += 1
        logging.info("CAM %d - GRAB - frame %d @ time %f", self.threadID,
                     self.framecounter, self.timestamp)

        # restart thread by unsuspending it
        self.suspend = False

        return ret_val

    def get(self, property_name):
        # get a video capture property

        # intercept calls to get the current timestamp or frame nunber
        # of the frame and explicitly return that of the last image
        # returned to the caller via read() or retrieve() from this object
        if (property_name == cv2.CAP_PROP_POS_MSEC):
            return self.timestamp_last_read
        elif (property_name == cv2.CAP_PROP_POS_FRAMES):
            return self.framecounter_last_read

        # default to behavior as per OpenCV manual for
        # VideoCapture()
        return self.camera.get(property_name)

    def getBackendName(self):
        # get a video capture backend (behavior as per OpenCV manual for
        # VideoCapture)
        return self.camera.getBackendName()

    def getExceptionMode(self):
        # get a video capture exception mode (behavior as per OpenCV manual for
        # VideoCapture)
        return self.camera.getExceptionMode()

    def setExceptionMode(self, enable):
        # get a video capture exception mode (behavior as per OpenCV manual for
        # VideoCapture)
        return self.camera.setExceptionMode(enable)

    def __del__(self):
        self.stopped = True
        self.suspend = True

    def __exit__(self, exec_type, exc_value, traceback):
        self.stopped = True
        self.suspend = True

##########################################################################