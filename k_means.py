from numbapro import cuda
from accelerate.cuda import rand
import numba
import numpy as np


def k_means_classify_seq(image, result, width, height, clusters, num_clusts, bands, itterations):
    k_means_group_seq(image, result, width, height, clusters, num_clusts, bands)
    for i in range(itterations - 1):
        k_means_cluster_seq(image, result, width, clusters, num_clusts, bands)
        k_means_group_seq(image, result, width, height, clusters, num_clusts, bands)

def k_means_cluster_seq(image, result, width, clusters, num_clusts, bands):
    new_clusters = np. np.ndarray([num_clust + 1, preGPU.shape[2]+1], dtype=np.uint16)
    for x in range(10):
        for y in range(width):
            for band in range(bands):
                new_clusters[result[x, y], band] += image[x, y, band]
            new_clusters[result[x, y], bands] += 1

    for cluster in range(1, num_clusts + 1):
        for band in range(bands):
            clusters[cluster, band] = int(host_clusters[cluster, band] / host_clusters[cluster, bands])


def k_means_group_seq(image, result, width, height, clusters, num_clusts, bands):
    for x in range(height):
        for y in range(width):
            datacheck = 1
            for band in range(bands):
                datacheck *= image[x, y, band]
            if datacheck != 0:
                dist = 0
                for band in range(bands):
                    dist += (image[x, y, band] - clusters[1, band]) ** 2
                min_dist = dist
                result[x, y] = 1
                for cluster in range(1, num_clusts + 1):
                    dist = 0
                    for band in range(bands):
                        dist += (image[x, y, band] - clusters[cluster, band]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        result[x, y] = cluster

                        # dev_result[x, y] = 1
            else:
                result[x, y] = 0



def k_means_classify(image, dev_result, x_size, y_size, clusters, num_clusts, bands, iterations, stream):
    new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1]+1], dtype=np.uint64, stream=stream)
    k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, y_size, clusters, num_clusts, bands)
    for i in range(iterations-1):
        #new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
        k_means_new_clusters[1, 1024, stream](image, dev_result, y_size, clusters, new_clusters, num_clusts, bands)
        k_means_move_clusters[1, min(num_clusts*bands, 1024), stream](clusters, new_clusters, num_clusts, bands)
        k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, y_size, clusters, num_clusts, bands)


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, uint16[:, :], int16, int16)', target='gpu')
def k_means_group(image, dev_result, width, clusters, num_clusts, bands):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    datacheck = 1
    for band in range(bands):
        datacheck *= image[x, y, band]
    if datacheck != 0:
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


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, uint16[:, :], uint64[:, :], int16, int16)', target='gpu')
def k_means_new_clusters(image, dev_result, width, clusters, new_clusters, num_clusts, bands):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    for band in range(bands):
        new_clusters[dev_result[x, y], band] += image[x, y, band]
    new_clusters[dev_result[x, y], bands] += 1


@cuda.jit('void(uint16[:, :], uint64[:, :], int16, int16)', target='gpu')
def k_means_move_clusters(clusters, new_clusters, num_clusters, bands):
    pix = cuda.grid(1)
    cluster = pix % num_clusters + 1
    band = pix / num_clusters

    clusters[cluster, band] = new_clusters[cluster, band] / new_clusters[cluster, bands]
    new_clusters[cluster, band] = 0
    new_clusters[cluster, bands] = 0


