scannertools
====================================

Scannertools is a high-level Python library for scalable video analysis built on the `Scanner <https://github.com/scanner-research/scanner/>`_ video processing engine. Scannertools provides easy-to-use, off-the-shelf implementations of:

* :ref:`object-detection`
* :ref:`face-detection`
* :ref:`pose-detection`
* :ref:`optical-flow`
* :ref:`shot-detection`
* :ref:`random-frame`

Examples are provided below (visuals from `this video <https://www.youtube.com/watch?v=_oLBVF_VYRM>`_). Auto-generated API documentation is `available here. <source/scannertools.html>`_

.. _object-detection:

Object detection
-------------------------------------
Find many kinds of objects (people, cars, chairs, ...) in a video using :func:`~scannertools.object_detection.detect_objects`. `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/object_detection.py>`__. ::

    import scannertools.object_detection as objdet
    import scannertools.vis as vis
    bboxes = objdet.detect_objects(db, video)
    vis.draw_bboxes(db, video, bboxes)

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/6xt-YVFCC9I" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


.. _face-detection:

Face detection
-------------------------------------
Find people's faces in a video using :func:`~scannertools.face_detection.detect_faces`.  `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/face_detection.py>`__. ::

    import scannertools.face_detection as facedet
    import scannertools.vis as vis
    bboxes = facedet.detect_faces(db, video)
    vis.draw_bboxes(db, video, bboxes)

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/IQsb_nbPf9M" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


.. _pose-detection:

Pose detection
-------------------------------------
Detect people's poses (two-dimensional skeletons) using :func:`~scannertools.pose_detection.detect_poses`. `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/pose_detection.py>`__. ::

    import scannertools.pose_detection as posedet
    import scannertools.vis as vis
    poses = posedet.detect_poses(db, video)
    vis.draw_poses(db, video, poses)

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/N1bT1yjnvMY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

.. _optical-flow:

Optical flow
-------------------------------------
Compute dense motion vectors between frames with :func:`~scannertools.optical_flow.compute_flow`. `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/optical_flow.py>`__. ::

    import scannertools.optical_flow as optflow
    import scannertools.vis as vis
    flow_fields = optflow.compute_flow(db, video)
    vis.draw_flow_fields(db, video, flow_fields)

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/ru048EWgc2Y" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

.. _shot-detection:

Shot detection
-------------------------------------
Find shot changes in a video using :func:`~scannertools.shot_detection.detect_shots`. `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/shot_detection.py>`__. ::

    import scannertools.shot_detection as shotdet
    shots = shotdet.detect_shots(db, video)
    montage_img = video.montage(shots)
    tb.imwrite('shots.jpg', montage_img)

.. image:: https://storage.googleapis.com/scanner-data/public/sample-shots-small.jpg
   :target: https://storage.googleapis.com/scanner-data/public/sample-shots.jpg

.. _random-frame:

Random frame access
-------------------------------------
Extract individual frames from a video with low overhead using :meth:`Video.frame <scannertools.video.Video.frame>`. `Full example here <https://github.com/scanner-research/scannertools/blob/master/examples/frame_montage.py>`__. ::

    frame = video.frame(0)
    tb.imwrite('frame0.jpg', frame)

.. image:: https://storage.googleapis.com/scanner-data/public/sample-frame.jpg
