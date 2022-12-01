import numpy as np
import cv2 as cv
from scipy.io import loadmat, savemat
import tifffile as tiff
import spectral
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# raw_data = loadmat('./HSIdata/rawdata/sv/Salinas_corrected.mat')['salinas_corrected']
# raw_data = loadmat('./HSIdata/rawdata/pu/PaviaU.mat')['paviaU']
raw_data = loadmat('./HSIdata/rawdata/dc/Washington_DC.mat')['washington_dc']
raw_data = loadmat('./HSIdata/rawdata/houston2018/Houston2018.mat')['houston2018']
# raw_data = raw_data.transpose([2, 0, 1])
print(raw_data.shape)
# tiff.imwrite('./HSIdata/rawdata/dc/washington_dc.tif', raw_data)
# raw_data = tiff.imread('./HSIdata/rawdata/dc/dc.tif')
# raw_data = raw_data.transpose([1, 2, 0])
# savemat('./HSIdata/rawdata/dc/Washington_DC_full.mat', {'washington_dc_full': raw_data})
# raw_data = loadmat('./HSIdata/rawdata/dc/Washington_DC_full.mat')['washington_dc_full']
# raw_data = tiff.imread('./HSIdata/rawdata/dc/dc.tif')
# raw_data = raw_data.transpose([1,2,0])
# gt = tiff.imread('./HSIdata/rawdata/dc/DC/GT.tif')
# print(raw.shape)
# print(gt.shape)
# print(np.unique(gt))
# spectral.save_rgb('./HSIdata/rawdata/dc/washington_dc_full.tif', raw, [60, 27, 17])
spectral.save_rgb('./HSIdata/rawdata/houston2018/houston2018_rgb.jpg', raw_data, [47, 32, 16])
# spectral.save_rgb('./HSIdata/rawdata/dc/washington_dc_full.jpg', raw_data, [131, 164, 174])
# raw_data = loadmat('./HSIdata/rawdata/hu/Houston2013.mat')['houston2013']
# raw_data = loadmat('./HSIdata/rawdata/ip/Indian_pines_corrected.mat')['indian_pines_corrected']
# gt_data = loadmat('./indian/Indian_pines_gt.mat')['indian_pines_gt']
# gt_data = loadmat('./HSIdata/rawdata/dc/Washington_DC_gt.mat')['washington_dc_gt']
# gt_data = loadmat('./HSIdata/rawdata/hu/Houston2013_gt.mat')['houston2013_gt']
# print(raw_data.shape)
# print(gt_data.shape)


reshape_data = raw_data.reshape((raw_data.shape[0]*raw_data.shape[1], raw_data.shape[2]))
pca = PCA(n_components=3)
pca.fit(reshape_data)
pca_data = pca.fit_transform(reshape_data)
print(pca_data.shape)
input_image = pca_data.reshape((raw_data.shape[0], raw_data.shape[1], 3))

# spectral.save_rgb('./HSIdata/rawdata/dc/washington_dc.jpg', raw_data, [29, 19, 9])
# spectral.save_rgb('./HSIdata/rawdata/dc/washington_dc.jpg', raw_data, [150, 130, 110])
# spectral.save_rgb('./HSIdata/rawdata/hu/houston2013.jpg', raw_data, [59, 40, 13])
# spectral.imshow(input_image)
# print(input_image)
# spectral.save_rgb('./HSIdata/rawdata/hu/houston2013_pca.jpg', input_image)
# spectral.save_rgb('./HSIdata/rawdata/dc/washington_dc_full_pca.jpg', input_image)
spectral.save_rgb('./HSIdata/rawdata/houston2018/Houston2018_pca.jpg', input_image)
# spectral.imshow(gt_data)
# spectral.view_cube(raw_data)

