# Tixelbox: video processing toolkit &nbsp; [![Build Status](https://travis-ci.org/scanner-research/tixelbox.svg?branch=master)](https://travis-ci.org/scanner-research/tixelbox)

Tixelbox is a high-level Python library for scalable video analysis built on the [Scanner](https://github.com/scanner-research/scanner/) video processing engine. Tixelbox provides easy-to-use, off-the-shelf implementations of:

* [Object detection](https://github.com/scanner-research/tixelbox/blob/master/examples/object_detection.py)
* [Face detection](https://github.com/scanner-research/tixelbox/blob/master/examples/face_detection.py)
* [Pose detection](https://github.com/scanner-research/tixelbox/blob/master/examples/pose_detection.py)
* [Optical flow](https://github.com/scanner-research/tixelbox/blob/master/examples/optical_flow.py)
* [Shot detection](https://github.com/scanner-research/tixelbox/blob/master/examples/shot_detection.py)
* [Random frame access](https://github.com/scanner-research/tixelbox/blob/master/examples/frame_montage.py)

See the [documentation](https://scanner-research.github.io/tixelbox/) for details.

## Usage

Here's an example using Tixelbox to extract faces in every 10th frame of a video, and then to draw the bounding boxes on one of those frames.

```python
import tixelbox as tb
import tixelbox.face_detection as facedet
import scannerpy
import cv2

# Get a reference to the video
video = tb.Video('path/to/your/video.mp4')
frame_nums = list(range(0, video.num_frames(), 10))

# Run the face detection algorithm
with scannerpy.Database() as db:
    face_bboxes = facedet.detect_faces(db, video, frame_nums)

# Draw the bounding boxes
frame = video.frame(frame_nums[3])
for bbox in face_bboxes[3]:
    cv2.rectangle(
        frame,
        (int(bbox.x1 * video.width()), int(bbox.y1 * video.height())),
        (int(bbox.x2 * video.width()), int(bbox.y2 * video.height())),
        (255, 0, 0),
        4)

# Save the image to disk
tb.imwrite('example.jpg', frame)
```

For more examples, see the [examples](https://github.com/scanner-research/tixelbox/tree/master/examples) directory. For the API reference, see our [documentation](https://scanner-research.github.io/tixelbox/).

## Installation

Tixelbox requires the Python packages for our three libraries [Scanner](https://github.com/scanner-research/scanner/), [Storehouse](https://github.com/scanner-research/storehouse/), and [Hwang](https://github.com/scanner-research/hwang) to be installed. To get these, you can either use our prebuilt Docker image for Scanner ([`scannerresearch/scanner`](https://hub.docker.com/r/scannerresearch/scanner/)), or you can follow the [installation instructions](https://github.com/scanner-research/scanner/blob/master/INSTALL.md) in the Scanner repository. Note that Tixelbox has optional dependencies for certain pipelines, namely TensorFlow and OpenCV.

With the dependencies installed, just run:

```
pip install tixelbox
```

Then try running any of the examples!
