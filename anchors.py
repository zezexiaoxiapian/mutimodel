import glob
import imagesize
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

def iou_distance_wh(wh1, wh2):
    wh1 = wh1 / 2
    wh2 = wh2 / 2
    inter = np.prod(np.minimum(wh1, wh2))
    union = np.prod(wh1) + np.prod(wh2) - inter
    return 1 - inter / (union + 1e-10)

sets=['train', 'val']
label_files = chain(*[glob.glob('/home/eleflea/datasets/FLIR/%s/txt/*.txt' % s) for s in sets])
ws = []
hs = []
for file in label_files:
    the_path = file.replace('.txt', '.jpeg').replace('txt', 'thermal_8_bit')
    iw, ih = imagesize.get(the_path)
    with open(file, 'r') as fr:
        for line in fr.readlines():
            if not line.strip():
                continue
            ann = line.split(' ')
            rw, rh = float(ann[3]), float(ann[4])
            ws.append(rw * iw)
            hs.append(rh * ih)
print(f'{len(ws)} bboxes')

samples = np.array(list(zip(ws, hs)))
for _ in range(1):
    sample = samples[np.random.choice(samples.shape[0], 20000, replace=False), :]
    metric = distance_metric(type_metric.USER_DEFINED, func=iou_distance_wh)
    initial_centers = kmeans_plusplus_initializer(sample, 9).initialize()
    # Create instance of K-Means algorithm with prepared centers.
    kmeans_instance = kmeans(sample, initial_centers, metric=metric)
    # Run cluster analysis and obtain results.
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = np.array(kmeans_instance.get_centers())
    # Visualize obtained results
    kmeans_visualizer.show_clusters(sample, clusters, final_centers, display=False)
    sccs = np.round(final_centers[np.argsort(np.prod(final_centers, axis=1))]).astype(np.int)
    print(sccs)
# [[ 15  17]
#  [ 16  35]
#  [ 35  29]
#  [ 24  61]
#  [ 61  47]
#  [ 41 102]
#  [105  78]
#  [ 77 170]
#  [179 158]]
plt.savefig('results/flir_anchors.png')