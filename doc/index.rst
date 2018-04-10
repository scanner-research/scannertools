.. tixelbox documentation master file, created by
   sphinx-quickstart on Sun Apr  1 22:41:48 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tixelbox
====================================

Tixelbox is a high-level Python library for scalable video analysis built on the Scanner video processing engine. Tixelbox is not an ffmpeg replacement--its primary purpose is analysis, not transcoding. Tixelbox provides easy-to-use, off-the-shelf implementations of various video processing algorithms:

* :ref:`object-detection`
* :ref:`face-detection`
* :ref:`pose-detection`
* :ref:`optical-flow`
* :ref:`shot-detection`
* :ref:`random-frame`

.. _object-detection:

Object detection
-------------------------------------
Find many kinds of objects (people, cars, chairs, ...) in a video using :func:`~tixelbox.object_detection.detect_objects`. `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/object_detection.py>`__.


.. _face-detection:

Face detection
-------------------------------------
Find people's faces in a video using :func:`~tixelbox.face_detection.detect_faces`.  `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/face_detection.py>`__.


.. _pose-detection:

Pose detection
-------------------------------------
Detect people's poses (two-dimensional skeletons) using :func:`~tixelbox.pose_detection.detect_poses`. `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/pose_detection.py>`__.


.. _optical-flow:

Optical flow
-------------------------------------
Compute dense motion vectors between frames with :func:`~tixelbox.optical_flow.compute_flow`. `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/optical_flow.py>`__.

.. _shot-detection:

Shot detection
-------------------------------------
Find shot changes in a video using :func:`~tixelbox.shot_detection.detect_shots`. `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/shot_detection.py>`__.

.. _random-frame:

Random frame access
-------------------------------------
Extract individual frames from a video with low overhead using :meth:`~tixelbox.video.Video.frame`. `Full example here <https://github.com/scanner-research/tixelbox/blob/master/examples/frame_montage.py>`__.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
