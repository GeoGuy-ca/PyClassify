from numbapro import cuda

@cuda.jit('void(int16[:,:,:], int8[:,:], int16, int16, int16)', target='gpu')
def k_means_classify(image, dev_result, width, clusters, max_iteration):
    x = cuda.grid(1)
    #dev_result[x, y] = image[x, y, 0]
    dev_result[x, 0] = x
