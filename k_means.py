from numbapro import cuda
import math
from numbapro import vectorize
import numba
import numpy as np

stop = False


def k_means_classify_seq(image, result, width, height, clusters, num_clusts, bands, itterations, debug):
    global stop
    k_means_group_seq(image, result, width, height, clusters, num_clusts, bands)
    for i in range(itterations - 1):
        k_means_cluster_seq(image, result, width, height, clusters, num_clusts, bands)
        if stop and not debug:
            print "Finished after " + str(i + 1) + " iterations."
            break
        k_means_group_seq(image, result, width, height, clusters, num_clusts, bands)


def k_means_classify_seq_no_root(image, result, width, height, clusters, num_clusts, bands, itterations, debug):
    global stop
    k_means_group_seq_no_root(image, result, width, height, clusters, num_clusts, bands)
    for i in range(itterations - 1):
        k_means_cluster_seq(image, result, width, height, clusters, num_clusts, bands)
        if stop and not debug:
            print "Finished after " + str(i + 1) + " iterations."
            break
        k_means_group_seq_no_root(image, result, width, height, clusters, num_clusts, bands)


def k_means_cluster_seq(image, result, width, height, clusters, num_clusts, bands):
    global stop
    stop = True
    new_clusters = np.zeros([num_clusts + 1, bands+1], dtype=np.uint64)
    for x in range(height):
        for y in range(width):
            for band in range(bands):
                new_clusters[result[x, y], band] += image[x, y, band]
            new_clusters[result[x, y], bands] += 1
    for cluster in range(1, num_clusts + 1):
        if new_clusters[cluster, bands] > 0:
            for band in range(bands):
                new_val = int(new_clusters[cluster, band] / new_clusters[cluster, bands])
                if clusters[cluster, band] != new_val:
                    clusters[cluster, band] = new_clusters[cluster, band] / new_clusters[cluster, bands]
                    stop = False
                new_clusters[cluster, band] = 0
            new_clusters[cluster, bands] = 0
        else:
            for band in range(bands):
                clusters[cluster, band] = 0


def k_means_group_seq(image, result, width, height, clusters, num_clusts, bands):
    for x in range(height):
        for y in range(width):
            datacheck = 1
            for band in range(bands):
                datacheck *= image[x, y, band]
            if datacheck != 0:
                dist = np.sqrt(sum(image[x, y, :] - clusters[1, band]))
                min_dist = dist
                result[x, y] = 1
                for cluster in range(1, num_clusts + 1):
                    dist = np.sqrt(sum(image[x, y, :] - clusters[cluster, band]))
                    if dist < min_dist:
                        min_dist = dist
                        result[x, y] = cluster
            else:
                result[x, y] = 0


def k_means_group_seq_fuzzy(image, result, width, height, clusters, num_clusts, bands, distance):
    for x in range(height):
        for y in range(width):
            datacheck = 1
            for band in range(bands):
                datacheck *= image[x, y, band]
            if datacheck != 0:
                distance[x, y, num_clusts + 1] = 0
                distance[x, y, 0] = np.sqrt(sum(image[x, y, :] - clusters[1, band]))
                distance[x, y, num_clusts + 1] += distance[x, y, 0]
                min_dist = distance[x, y, 0]
                result[x, y] = 1
                for cluster in range(1, num_clusts + 1):
                    distance[x, y, cluster] = np.sqrt(sum(image[x, y, :] - clusters[cluster, band]))
                    distance[x, y, num_clusts + 1] += distance[x, y, cluster]
                    if distance[x, y, cluster] < min_dist:
                        min_dist = distance[x, y, cluster]
                        result[x, y] = cluster
            else:
                result[x, y] = 0


def k_means_cluster_seq_fuzzy(image, result, width, height, clusters, num_clusts, bands, distance, movement):
    global stop
    stop = True
    new_clusters = np.zeros([num_clusts + 1, bands+1], dtype=np.float64)
    for x in range(height):
        for y in range(width):
            for cluster in range(num_clusts+1):
                weight = 1.0 / (distance[x, y, num_clusts + 1] / (distance[x, y, num_clusts + 1] - distance[x, y, cluster]))
                for band in range(bands):
                    new_clusters[result[x, y], band] += image[x, y, band] * weight
                new_clusters[result[x, y], bands] += weight
    for cluster in range(num_clusts + 1):
        if new_clusters[cluster, bands] > 0:
            for band in range(bands):
                new_val = int(new_clusters[cluster, band] / new_clusters[cluster, bands])
                if abs(clusters[cluster, band] - new_val) > movement:
                    stop = False
                clusters[cluster, band] = new_clusters[cluster, band] / new_clusters[cluster, bands]
                new_clusters[cluster, band] = 0
            new_clusters[cluster, bands] = 0
        else:
            for band in range(bands):
                clusters[cluster, band] = 0


def k_means_group_seq_no_root(image, result, width, height, clusters, num_clusts, bands):
    for x in range(height):
        for y in range(width):
            datacheck = 1
            for band in range(bands):
                datacheck *= image[x, y, band]
            if datacheck != 0:
                dist = sum(image[x, y, :] - clusters[1, band])
                min_dist = dist
                result[x, y] = 1
                for cluster in range(1, num_clusts + 1):
                    dist = sum(image[x, y, :] - clusters[cluster, band])
                    if dist < min_dist:
                        min_dist = dist
                        result[x, y] = cluster
            else:
                result[x, y] = 0


def k_means_classify_fuzzy_seq(image, result, width, height, clusters, num_clusts, bands, itterations, movement, debug):
    global stop
    distances = np.ndarray([height, width, num_clusts + 2], dtype=np.float32)
    k_means_group_seq_fuzzy(image, result, width, height, clusters, num_clusts, bands, distances)
    for i in range(itterations - 1):
        k_means_cluster_seq_fuzzy(image, result, width, height, clusters, num_clusts, bands, distances, movement)
        if stop and not debug:
            print "Finished after " + str(i + 1) + " iterations."
            break
        k_means_group_seq_fuzzy(image, result, width, height, clusters, num_clusts, bands, distances)


def k_means_classify(image, dev_result, x_size, y_size, clusters, num_clusts, bands, iterations, debug, stream):
    if not debug:
        stop = np.ndarray(1, np.bool_)
        dev_stop = cuda.device_array(1, np.bool_)
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1]+1], dtype=np.uint64, stream=stream)
        k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)
        for i in range(iterations-1):
            #new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands)
            k_means_move_clusters[1, min(num_clusts*bands, 1024), stream](clusters, new_clusters, num_clusts, bands, dev_stop)
            dev_stop.copy_to_host(stop, stream=stream)
            if stop[0] and not debug:
                print "Finished after " + str(i + 1) + " iterations."
                break
            k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)

    else:
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1] + 1], dtype=np.uint64, stream=stream)
        k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)
        for i in range(iterations - 1):
            # new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands)
            k_means_move_clusters_no_check[1, min(num_clusts * bands, 1024), stream](clusters, new_clusters, num_clusts, bands)
            k_means_group[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)


def k_means_classify_no_root(image, dev_result, x_size, y_size, clusters, num_clusts, bands, iterations, debug, stream):
    if not debug:
        stop = np.ndarray(1, np.bool_)
        dev_stop = cuda.device_array(1, np.bool_)
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1]+1], dtype=np.uint64, stream=stream)
        k_means_group_no_root[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)
        for i in range(iterations-1):
            #new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands)
            k_means_move_clusters[1, min(num_clusts*bands, 1024), stream](clusters, new_clusters, num_clusts, bands, dev_stop)
            dev_stop.copy_to_host(stop, stream=stream)
            if stop[0] and not debug:
                print "Finished after " + str(i + 1) + " iterations."
                break
            k_means_group_no_root[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)

    else:
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1] + 1], dtype=np.uint64, stream=stream)
        k_means_group_no_root[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands)
        for i in range(iterations - 1):
            # new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands)
            k_means_move_clusters_no_check[1, min(num_clusts * bands, 1024), stream](clusters, new_clusters, num_clusts, bands)
            k_means_group_no_root[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size,
                                                                              clusters, num_clusts, bands)


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, int16, uint16[:, :], int16, int16)', target='gpu')
def k_means_group(image, dev_result, width, height, clusters, num_clusts, bands):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    #if x < height and y < width:
    datacheck = 1
    for band in range(bands):
        datacheck *= image[x, y, band]
    if datacheck != 0:
        dist = 0
        for band in range(bands):
            dist += (image[x, y, band] - clusters[1, band]) ** 2
        min_dist = math.sqrt(dist)
        dev_result[x, y] = 1
        for cluster in range(1, num_clusts + 1):
            dist = 0
            for band in range(bands):
                dist += (image[x, y, band] - clusters[cluster, band]) ** 2
            dist = math.sqrt(dist)
            if dist < min_dist:
                min_dist = dist
                dev_result[x, y] = cluster

        # dev_result[x, y] = 1
    else:
        dev_result[x, y] = 0


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, int16, uint16[:, :], int16, int16)', target='gpu')
def k_means_group_no_root(image, dev_result, width, height, clusters, num_clusts, bands):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    #if x < height and y < width:
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
            dist = dist
            if dist < min_dist:
                min_dist = dist
                dev_result[x, y] = cluster

        # dev_result[x, y] = 1
    else:
        dev_result[x, y] = 0


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, int16, uint16[:, :], uint64[:, :], int16, int16)', target='gpu')
def k_means_new_clusters(image, dev_result, width, height, clusters, new_clusters, num_clusts, bands):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    if x < width and y < height:
        for band in range(bands):
            new_clusters[dev_result[x, y], band] += image[x, y, band]
        new_clusters[dev_result[x, y], bands] += 1


@cuda.jit('void(uint16[:, :], uint64[:, :], int16, int16, bool_[:])', target='gpu')
def k_means_move_clusters(clusters, new_clusters, num_clusters, bands, stop):
    pix = cuda.grid(1)
    cluster = pix % num_clusters + 1
    band = pix / num_clusters
    stop[0] = True
    cuda.syncthreads()

    new_val = int(new_clusters[cluster, band] / new_clusters[cluster, bands])
    if clusters[cluster, band] != new_val:
        clusters[cluster, band] = new_val
        stop[0] = False
    new_clusters[cluster, band] = 0
    new_clusters[cluster, bands] = 0


@cuda.jit('void(uint16[:, :], uint64[:, :], int16, int16)', target='gpu')
def k_means_move_clusters_no_check(clusters, new_clusters, num_clusters, bands):
    pix = cuda.grid(1)
    cluster = pix % num_clusters + 1
    band = pix / num_clusters

    clusters[cluster, band] = new_clusters[cluster, band] / new_clusters[cluster, bands]
    new_clusters[cluster, band] = 0
    new_clusters[cluster, bands] = 0


def k_means_classify_fuzzy(image, dev_result, x_size, y_size, clusters, num_clusts, bands, iterations, debug, movement, stream):
    dev_distance = cuda.device_array([y_size, x_size, (num_clusts + 1)], dtype=np.float64, stream=stream)
    if not debug:
        stop = np.ndarray(1, np.bool_)
        dev_stop = cuda.device_array(1, np.bool_)
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1]+1], dtype=np.float32, stream=stream)
        k_means_group_fuzzy[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands, dev_distance)
        for i in range(iterations-1):
            #new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters_fuzzy[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands, dev_distance)
            k_means_move_clusters_fuzzy[1, min(num_clusts*bands, 1024), stream](clusters, new_clusters, num_clusts, bands, dev_stop, movement)
            dev_stop.copy_to_host(stop, stream=stream)
            if stop[0] and not debug:
                print "Finished after " + str(i + 1) + " iterations."
                break
            k_means_group_fuzzy[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands, dev_distance)
    else:
        new_clusters = cuda.device_array([clusters.shape[0], clusters.shape[1] + 1], dtype=np.uint64, stream=stream)
        k_means_group_fuzzy[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands, dev_distance)
        for i in range(iterations - 1):
            # new clusters must be done on a single block to maintain automicity only 1024 threads allowed :(
            k_means_new_clusters_fuzzy[1, 1024, stream](image, dev_result, x_size, y_size, clusters, new_clusters, num_clusts, bands, dev_distance)
            k_means_move_clusters_no_check[1, min(num_clusts * bands, 1024), stream](clusters, new_clusters, num_clusts, bands)
            k_means_group_fuzzy[(x_size * y_size) / 1024 + 1, 1024, stream](image, dev_result, x_size, y_size, clusters, num_clusts, bands, dev_distance)


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, int16, uint16[:, :], int16, int16, float32[:, :, :])', target='gpu')
def k_means_group_fuzzy(image, dev_result, width, height, clusters, num_clusts, bands, distance):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    #if x < height and y < width:
    datacheck = 1
    for band in range(bands):
        datacheck *= image[x, y, band]
    if datacheck != 0:
        distance[x, y, num_clusts + 1] = 0
        distance[x, y, 0] = 0
        for band in range(bands):
            distance[x, y, 0] += (image[x, y, band] - clusters[1, band]) ** 2
        #distance[x, y, 0] = math.sqrt(distance[x, y, 0])
        distance[x, y, num_clusts + 1] += distance[x, y, 0]
        min_dist = distance[x, y, 0]
        dev_result[x, y] = 1
        for cluster in range(1, num_clusts + 1):
            distance[x, y, cluster] = 0
            for band in range(bands):
                distance[x, y, cluster] += (image[x, y, band] - clusters[cluster, band]) ** 2
            #distance[x, y, cluster] = math.sqrt(distance[x, y, cluster])
            distance[x, y, num_clusts + 1] += distance[x, y, cluster]
            if distance[x, y, cluster] < min_dist:
                min_dist = distance[x, y, cluster]
                dev_result[x, y] = cluster

        # dev_result[x, y] = 1
    else:
        dev_result[x, y] = 0


@cuda.jit('void(uint16[:,:,:], uint8[:, :], int16, int16, uint16[:, :], float32[:, :], int16, int16, float32[:, :, :])', target='gpu')
def k_means_new_clusters_fuzzy(image, dev_result, width, height, clusters, new_clusters, num_clusts, bands, distance):
    pix = cuda.grid(1)
    y = pix % width
    x = pix / width

    if x < width and y < height:
        for cluster in range(num_clusts + 1):
            weight = 1.0 / (distance[x, y, num_clusts + 1] / (distance[x, y, num_clusts + 1] - distance[x, y, cluster]))
            for band in range(bands):
                new_clusters[cluster, band] += image[x, y, band] * weight
            new_clusters[cluster, bands] += weight

@cuda.jit('void(uint16[:, :], float32[:, :], int16, int16, bool_[:], float32)', target='gpu')
def k_means_move_clusters_fuzzy(clusters, new_clusters, num_clusters, bands, stop, movement):
    pix = cuda.grid(1)
    cluster = pix % num_clusters + 1
    band = pix / num_clusters
    stop[0] = True
    cuda.syncthreads()

    new_val = int(new_clusters[cluster, band] / new_clusters[cluster, bands])
    if abs(clusters[cluster, band] - new_val) > movement:
        stop[0] = False
    clusters[cluster, band] = new_val
    new_clusters[cluster, band] = 0
    new_clusters[cluster, bands] = 0