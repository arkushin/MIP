import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import copy

SEEDS_NUM = 200
LIVER_MIN_TH = -100
LIVER_MAX_TH = 200


# todo: mention that all files should be in the same directory as ex3.py


def segmentLiver(ctFileName, AortaFileName, outputFileName):
    """
    A function that receives the names of the aorta and CT files it should use, and segments the liver in the original CT
    The function saves and returns a segmentation file called 'outputFileName'
    """
    # ct_scan = nib.load(ctFileName)
    # ct_data = ct_scan.get_data()
    file_name = ctFileName.split('.')[0] #todo: maybe remove

    orientation_flags = img_orientation(ctFileName)


    ROI_segmentation = find_ROI(ctFileName, AortaFileName)
    multipleSeedsRG(ctFileName, ROI_segmentation)

    ct_scan = nib.load(file_name + '_region_growing.nii.gz')
    ct_data = ct_scan.get_data()
    #
    for slc in range(ct_scan.shape[2]):
        if ct_data[:,:,slc].any():
            ct_data[:,:,slc] = morphology.remove_small_holes(ct_data[:,:,slc].astype(np.uint8), area_threshold=200)
            ct_data[:,:,slc], connected_components_num = measure.label(ct_data[:,:,slc], return_num=True)
            props = measure.regionprops(ct_data[:,:,slc].astype(np.uint16), coordinates='rc')
            area_list = [region.area for region in props]
            ct_data[:,:,slc] = morphology.remove_small_objects(ct_data[:,:,slc].astype(np.uint8), min_size=max(area_list))

    ct_data[ct_data != 0] = 1

    # save the segmentation of the liver as a nifti file
    nib.save(ct_scan, outputFileName + '.nii.gz')



def multipleSeedsRG(ctFileName, ROI_segmentation):
    """
    A function that executes multiple seeded region growing for the given CT, based on seeds selected from the given ROI
    :return The function returns the resulting segmentation of the liver, with no morphological operation performed yet
    """
    print('start: region_growing')
    file_name = ctFileName.split('.')[0] #todo: maybe remove

    ct_img = nib.load(ctFileName)
    ct_data = ct_img.get_data()
    seeds_seg = nib.load(ctFileName)
    seeds_data = seeds_seg.get_data()

    seeds_list = find_seeds(ctFileName, ROI_segmentation)


    # convert seeds list to segmentation and save it
    seeds_data[::] = 0
    for seed in seeds_list:
        seeds_data[seed[0], seed[1], seed[2]] = 1

    nib.save(seeds_seg, file_name + '_seeds_seg.nii.gz')  # todo: maybe disable this line
    print('seeds data saved')

    # perform seeded region growing:
    cube = morphology.cube(3)
    last_region = copy.deepcopy(seeds_data).astype(np.uint8)
    cur_region = morphology.dilation(last_region, cube).astype(np.uint8)

    last_region_num = np.sum(last_region)
    cur_region_num = np.sum(cur_region)
    i = 0
    # while last_region_num != cur_region_num:
    for i in range(100):
        print(i)
        region_mean = np.mean(ct_data[last_region == 1])
        neighbors_flags = np.zeros(ct_data.shape)
        neighbors_indexes = np.subtract(cur_region, last_region)
        neighbors_flags[neighbors_indexes == 1] = ct_data[neighbors_indexes == 1]
        neighbors_flags[np.abs(neighbors_flags - region_mean) <= 20] = 1
        neighbors_flags[neighbors_flags != 1] = 0

        last_region = np.logical_or(neighbors_flags, last_region).astype(np.uint8)

        if i % 2:
            cur_region_num = np.sum(last_region)
        else:
            last_region_num = np.sum(last_region)

        cur_region = morphology.binary_dilation(last_region, cube).astype(np.uint8)

        if not i % 10:
            print(cur_region_num)

        i += 1

    print(last_region_num)
    seeds_data[:, :, :] = 0
    seeds_data[last_region == 1] = 1

    nib.save(seeds_seg, file_name + '_region_growing.nii.gz')
    print('region growing saved')
    print('end: region_growing')
    # return seeds_data  # todo: check! was seeds_seg before


def find_seeds(ctFileName, ROI_segmentation):
    """
    A function that receives a CT scan and an ROI segmentation of the CT and returns a list of 200 seeds that are
    located within the liver
    """
    print('start: find_seeds')
    ct_img = nib.load(ctFileName)
    ct_data = ct_img.get_data()
    # ROI_img = nib.load(ROI_segmentation)
    ROI_data = ROI_segmentation

    seeds = []

    # find the borders of the ROI:
    upper_z = max([z if ROI_data[:, :, z].any() else 0 for z in range(ROI_data.shape[2])])
    lower_z = min([z if ROI_data[:, :, z].any() else 512 for z in range(ROI_data.shape[2])])
    upper_x = max([row if ROI_data[:, row, :].any() else 0 for row in range(ROI_data.shape[1])])
    lower_x = min([row if ROI_data[:, row, :].any() else 512 for row in range(ROI_data.shape[1])])
    left_y = max([col if ROI_data[col, :, :].any() else 0 for col in range(ROI_data.shape[0])])
    right_y = min([col if ROI_data[col, :, :].any() else 512 for col in range(ROI_data.shape[0])])

    # randomly sample points in the ROI and validate that they are in the liver
    while len(seeds) < SEEDS_NUM:
        z = np.random.randint(lower_z, upper_z + 1)
        x = np.random.randint(lower_x, upper_x + 1)
        y = np.random.randint(right_y + 50, left_y + 1)

        if LIVER_MIN_TH < ct_data[y, x, z] < LIVER_MAX_TH and ROI_data[y,x,z]:
            seeds.append([y, x, z])

    print('end: find_seeds')


    return seeds


def find_ROI(ctFileName, AortaFileName):
    """
    A function that finds an ROI of the liver for the given CT file using the segmentation of the aorta.
    :return the function returns a segmentation file in format nifti of the CT such that all pixels in the ROI are 1
    and the rest are 0
    """
    print('start: find_ROI')
    file_name = ctFileName.split('.')[0]


    ct_img = nib.load(ctFileName)
    ct_data = ct_img.get_data()
    body_seg = IsolateBody(ctFileName)
    # body_seg = nib.load('Case1_CT_bodySeg.nii.gz')  # todo: return use in IsolateBody
    body_data = body_seg.get_data()
    aorta_seg = nib.load(AortaFileName)
    aorta_data = aorta_seg.get_data()

    # find borders of aorta:
    lower_border = 0
    while not aorta_data[:, :, lower_border].any():
        lower_border += 1
    upper_border = lower_border
    while aorta_data[:, :, upper_border].any():
        upper_border += 1

    aorta_mid = int((upper_border + lower_border) / 2)

    # find ROI borders
    ROI_upper = max([row if aorta_data[:, row, aorta_mid].any() else 0 for row in range(aorta_data.shape[1])])
    ROI_left = max([col if aorta_data[col, :, aorta_mid].any() else 0 for col in range(aorta_data.shape[0])])
    ROI_lower = min([row if body_data[:, row, aorta_mid].any() else 512 for row in range(body_data.shape[1])])
    ROI_right = max([col if body_data[col, :, aorta_mid].any() else 0 for col in range(body_data.shape[0])])

    # find the outlines of the skin, for the slice of interest:
    derivation_matrix = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    dx = convolve2d(body_data[:, :, aorta_mid], derivation_matrix, mode='same', boundary='wrap')
    dy = convolve2d(body_data[:, :, aorta_mid], derivation_matrix.T, mode='same', boundary='wrap')
    skin_outline = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)  # calculates the magnitude of dx and dy
    disk = morphology.disk(60)  # todo: check on more cases, maybe make a larger disk
    skin_outline = morphology.dilation(skin_outline, disk)
    skin_outline[skin_outline != 0] = 1
    skin_segmentation = np.zeros(ct_data.shape)
    skin_segmentation[:, :, aorta_mid] = skin_outline

    # create the ROI = segmentation file with 1's in the ROI and 0's in the rest of the pixels
    ROI_segmentation = np.zeros(ct_data.shape)
    ROI_segmentation[ROI_left:ROI_right, ROI_lower:ROI_upper, aorta_mid] = 1
    ROI_segmentation = np.logical_and(ROI_segmentation, body_data)
    ROI_segmentation = np.subtract(ROI_segmentation, skin_segmentation)

    ct_data[:, :, :] = 0
    ct_data[ROI_segmentation == 1] = 1

    nib.save(ct_img, file_name + '_ROI.nii.gz')
    print('ROI_saved')
    print('end: find_ROI')

    return ct_data


def IsolateBody(CT_scan):
    """
    A function that segments the body in the given CT
    :param CT_scan: The path to the CT scan that should be segmented
    :return The function returns the segmentation of the body
    """
    print('start: isolate_body')
    file_name = CT_scan.split('.')[0] #todo: delete before submission!

    img = nib.load(CT_scan)
    img_data = img.get_data()

    img_data[img_data == 0] = 1
    img_data[img_data < -500] = 0
    img_data[img_data > 2000] = 0
    img_data[img_data != 0] = 1

    # find largest connectivity component and remove all others:
    img_data[::], new_connected_components_num = measure.label(img_data, return_num=True)
    props = measure.regionprops(img_data.astype(np.uint16))
    max_area = props[0].area
    for i in range(1, new_connected_components_num):
        if props[i].area > max_area:
            max_area = props[i].area
    img_data[::] = morphology.remove_small_objects(img_data.astype(np.uint16), max_area)

    nib.save(img, file_name + '_bodySeg.nii.gz')
    print('bodySeg saved')
    print('end: isolate_body')
    return img


def flip_axis(nii_data, orientation_flags):
    """
    A function that flips the orientation of the axes according to the given orientation flags
    """
    for i in range(len(orientation_flags)):
        if orientation_flags[i]:
            nii_data = np.flip(nii_data, axis=i)
    return nii_data


def img_orientation(ctFileName):
    """
    A function that determines the orientation of the given CT scan. The code handles images in 'R,P,S' orientation
    """



if __name__ == '__main__':
    ctFileName = 'Case1_CT.nii.gz'
    AortaFileName = 'Case1_Aorta.nii.gz'
    segmentLiver(ctFileName, AortaFileName, 'Case1_segmented_liver')

