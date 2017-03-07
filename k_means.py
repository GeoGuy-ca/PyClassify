from numbapro import cuda
from accelerate.cuda import rand
import math

@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, uint16[:, :], int16, int16, int16)', target='gpu')
def k_means_classify(image, dev_result, width, clusters, num_clusts, bands, max_iteration):
    pix = cuda.grid(1)
    x = pix % width
    y = pix / width
    if image[x, y, 0] != 0:
        min_dist = 1000000
        for cluster in range(1, num_clusts + 1):
            dist = 0
            for band in range(bands):
                dist += (image[x, y, band] - clusters[cluster, band]) ** 2
            if dist < min_dist:
                dev_result[x, y] = cluster
                min_dist = dist