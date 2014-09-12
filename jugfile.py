import mahotas as mh
import numpy as np
from glob import glob
from jug import TaskGenerator

def reference_for(fname):
    return fname.replace('/input/','/references/').replace('.jpg', '.png')

@TaskGenerator
def segment(fname):
    dna = mh.imread(fname)
    dna = dna[:,:,0]

    sigma = 12.
    dnaf = mh.gaussian_filter(dna, sigma)

    T_mean = dnaf.mean()
    bin_image = dnaf > T_mean
    labeled, nr_objects = mh.label(bin_image)

    maxima = mh.regmax(mh.stretch(dnaf))
    maxima = mh.dilate(maxima, np.ones((5,5)))
    maxima,_ = mh.label(maxima)
    dist = mh.distance(bin_image)
    dist = 255 - mh.stretch(dist)
    watershed = mh.cwatershed(dist, maxima)
    watershed *= bin_image
    return watershed

@TaskGenerator
def evaluate(fname, output):
    from sklearn.metrics import cluster
    refname = reference_for(fname)
    ref = mh.imread(refname)
    return cluster.adjusted_rand_score(ref.ravel(), output.ravel())


images = glob('data/input/*.jpg')
results = []
segmented = []
for fname in images:
    attempt = segment(fname)
    results.append(evaluate(fname, attempt))
    segmented.append(attempt)

