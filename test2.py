import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import copy


ct_img = nib.load('data/Case1_CT.nii.gz')
ct_data = ct_img.get_data()
slc = copy.deepcopy(ct_data[:,:,244])

seeds_seg = nib.load('seeds_seg.nii.gz')
seeds_data_orig = seeds_seg.get_data()
seeds_data = copy.deepcopy(seeds_data_orig[:,:,244])

square = morphology.square(3)


last_region = copy.deepcopy(seeds_data)
cur_region = morphology.dilation(last_region, square)

last_region_num = np.sum(last_region)
cur_region_num = np.sum(cur_region)
i = 0
while last_region_num != cur_region_num:
# for i in range(1000):
    print(i)
    region_mean = np.mean(slc[last_region == 1])
    neighbors_flags = np.zeros(slc.shape)
    neighbors_indexes = np.subtract(cur_region, last_region)
    neighbors_flags[neighbors_indexes==1] = slc[neighbors_indexes==1]
    neighbors_flags[np.abs(neighbors_flags - region_mean) <= 20] = 1
    neighbors_flags[neighbors_flags != 1] = 0

    last_region = np.logical_or(neighbors_flags, last_region).astype(np.uint8)
    if i%2:
        cur_region_num = np.sum(last_region)
    else:
        last_region_num = np.sum(last_region)
    cur_region = morphology.dilation(last_region, square)

    i += 1

print(last_region_num)
cur_region = morphology.remove_small_holes(cur_region)
print(np.sum(cur_region))
plt.imshow(cur_region, cmap='gray')
plt.show()



