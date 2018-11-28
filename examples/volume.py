from scannertools import sample_video, audio
import scannerpy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with sample_video(delete=False) as video:
    db = scannerpy.Database()

    aud = audio.AudioSource(video)
    out = audio.compute_average_volume(db, audio=[aud])

    plt.plot(list(out[0].load()))
    plt.savefig('volume.png')
