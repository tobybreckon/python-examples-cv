# Python Computer Vision OpenCV Teaching Examples

OpenCV Python computer vision examples used for teaching within the undergraduate Computer Science programme
at [Durham University](http://www.durham.ac.uk) (UK) by [Prof. Toby Breckon](http://community.dur.ac.uk/toby.breckon/).

All tested with [OpenCV](http://www.opencv.org) 3.x and Python 3.x.

```
# Example : <................................> processing from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera
```
---

### Background:

Directly adapted (and in some cases extended/added to) from the [C++](https://github.com/tobybreckon/cpp-examples-ipcv.git) and earlier [C](https://github.com/tobybreckon/cpp-examples-ipcv.git) language teaching examples used to generate the video examples within the ebook version of:

[Dictionary of Computer Vision and Image Processing](http://dx.doi.org/10.1002/9781119286462) (R.B. Fisher, T.P. Breckon, K. Dawson-Howe, A. Fitzgibbon, C. Robertson, E. Trucco, C.K.I. Williams), Wiley, 2014.
[[Google Books](http://books.google.co.uk/books?id=TaEQAgAAQBAJ&lpg=PP1&dq=isbn%3A1118706811&pg=PP1v=onepage&q&f=false)] [[doi](http://dx.doi.org/10.1002/9781119286462)]

Notably, the [C++](https://github.com/tobybreckon/cpp-examples-ipcv.git) examples may contain further speed optimizations in some cases.

A related supporting set of [Python Image Processing OpenCV Teaching Examples](https://github.com/tobybreckon/python-examples-ip.git) are also available covering basic image processing operations.

---

### How to download and run:

Download each file as needed or to download the entire repository and run each try:

```
git clone https://github.com/tobybreckon/python-examples-cv.git
cd python-examples-cv
python3 ./<insert file name of one of the examples>.py [optional video file]
```

Demo source code is provided _"as is"_ to aid learning and understanding of topics on the course and beyond.

Most run with a webcam connected or from a command line supplied video file of a format OpenCV supports on your system (otherwise edit the script to provide your own image source).

N.B. you may need to change the line near the top that specifies the camera device to use on some examples below - change "0" if you have one webcam, I have it set to "1" to skip my built-in laptop webcam and use the connected USB camera.

---

If referencing these examples in your own work please use:
```
@Book{solomonbreckon10fundamentals,
  author = 	 {Solomon, C.J. and Breckon, T.P.},
  title = 	 {Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab},
  publisher = 	 {Wiley-Blackwell},
  year = 	 {2010},
  isbn = {0470844736},
  comment =   {<a class="demolink" href="http://www.fundipbook.com/">book website</a>},
  doi = 	 {10.1002/9780470689776},
  note = {ISBN-13: 978-0470844731},
  category =     {textbook}
}
```

---

If you find any bugs raise an issue (or much better still submit a git pull request with a fix) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
