from numbapro import cuda
from accelerate.cuda import rand
import math

@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, uint16[:, :], int16, int16, int16)', target='gpu')
def k_means_classify(image, dev_result, width, clusters, num_clusts, bands, max_iteration):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width
    if image[x, y, 0] != 0:
        dist = 0
        for band in range(bands):
            dist += (image[x, y, band] - clusters[1, band]) ** 2
        min_dist = dist
        dev_result[x, y] = 1
        for cluster in range(1, num_clusts + 1):
            dist = 0
            for band in range(bands):
                dist += (image[x, y, band] - clusters[cluster, band]) ** 2
            if dist < min_dist:
                min_dist = dist
                dev_result[x, y] = cluster

        # dev_result[x, y] = 1
    else:
        dev_result[x, y] = 0