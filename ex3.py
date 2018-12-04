import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import copy



def IsolateBody(CT_scan):
    """
    A function that segments the body in the given CT
    :param CT_scan: The path to the CT scan that should be segmented
    :return: The segmentation of the body
    """
    file_name = CT_scan.split('.')[0]

    img = nib.load(CT_scan)
    img_data = img.get_data()

    img_data[img_data == 0] = 1
    img_data[img_data < -500] = 0
    img_data[img_data > 2000] = 0
    img_data[img_data != 0] = 1

    img_data[::], new_connected_components_num = measure.label(img_data, return_num=True)
    props = measure.regionprops(img_data.astype(np.uint16))
    max_area = props[0].area
    for i in range(1, new_connected_components_num):
        if props[i].area > max_area:
            max_area = props[i].area
    img_data[::] = morphology.remove_small_objects(img_data.astype(np.uint16), max_area)

    nib.save(img, file_name + '_bodySeg.nii.gz')





if __name__ == '__main__':
    CT_scan = 'data/Case1_CT.nii.gz'
    IsolateBody(CT_scan)

