from numbapro import cuda

@cuda.jit('void(int16[:,:,:], int8[:,:], int16, int16)', target='gpu')
def k_means_classify(image, dev_result, clusters, max_iteration):
    for i in image[:, 0, 0]:
        for j in image[0, :, 0]:
            dev_result[i, j] = image[i, j, 0]
