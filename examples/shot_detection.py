import scannertools as st
import scannertools.shot_detection as shotdet
import scannerpy
import os

with st.sample_video() as video:
    db = scannerpy.Database()
    shots = shotdet.detect_shots(db, video)
    montage_img = video.montage(shots)
    st.imwrite('sample_shots.jpg', montage_img)
    print('Wrote shot montage to {}'.format(os.path.abspath('sample_shots.jpg')))
