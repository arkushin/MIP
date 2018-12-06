import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

SEEDS_NUM = 200
LIVER_MIN_TH = -100
LIVER_MAX_TH = 200


def IsolateBody(CT_scan):
    """
    A function that segments the body in the given CT
    :param CT_scan: The path to the CT scan that should be segmented
    :return The function returns the segmentation of the body
    """
    # file_name = CT_scan.split('.')[0] #todo: delete before submission!

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

    # nib.save(img, file_name + '_bodySeg.nii.gz')
    return img


def segmentLiver(ctFileName, AortaFileName, outputFileName):
    """
    A function that receives the names of the aorta and CT files it should use, and segments the liver in the original CT
    The function saves and returns a segmentation file called 'outputFileName'
    """
    ROI = find_ROI(ctFileName, AortaFileName)


def find_ROI(ctFileName, AortaFileName):
    """
    A function that finds an ROI of the liver for the given CT file using the segmentation of the aorta.
    :return the function returns segmentation of the CT such that all pixels in the ROI are 1 and the rest are 0
    """
    ct_img = nib.load(ctFileName)
    ct_data = ct_img.get_data()
    # body_seg = IsolateBody(ctFileName)
    body_seg = nib.load('data/Case1_CT_bodySeg.nii.gz')  # todo: return use in IsolateBody
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
    # print('aorta mid:', aorta_mid)

    # find ROI borders
    ROI_upper = max([row if aorta_data[:, row, aorta_mid].any() else 0 for row in range(aorta_data.shape[1])])
    ROI_left = max([col if aorta_data[col, :, aorta_mid].any() else 0 for col in range(aorta_data.shape[0])])
    ROI_lower = min([row if body_data[:, row, aorta_mid].any() else 512 for row in range(body_data.shape[1])])
    ROI_right = max([col if body_data[col, :, aorta_mid].any() else 0 for col in range(body_data.shape[0])])

    # print('left:', ROI_left)
    # print('right:', ROI_right)
    # print('lower:', ROI_lower)
    # print('upper:', ROI_upper)
    #
    # ROI = ct_data[ROI_left:ROI_right, ROI_lower:ROI_upper, aorta_mid]

    # plt.imshow(ROI.T, cmap='gray')
    # plt.title('ROI')
    # plt.show()
    # Find the outlines of the skin:
    conv_matrix = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    dx = convolve2d(body_data[:, :, aorta_mid], conv_matrix, mode='same', boundary='wrap')
    dy = convolve2d(body_data[:, :, aorta_mid], conv_matrix.T, mode='same', boundary='wrap')
    skin_outline = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    disk = morphology.disk(35)   # todo: check on more cases, maybe make a larger disk
    skin_outline = morphology.dilation(skin_outline, disk)
    skin_outline[skin_outline != 0] = 1

    skin_segmentation = np.zeros(ct_data.shape)
    skin_segmentation[:, :, aorta_mid] = skin_outline

    # plt.imshow(skin_outline.T, cmap='gray')
    # plt.show()

    ROI_as_seg = np.zeros(ct_data.shape)
    ROI_as_seg[ROI_left:ROI_right, ROI_lower:ROI_upper, aorta_mid] = 1

    ROI_as_seg = np.logical_and(ROI_as_seg, body_data)
    # plt.imshow(ROI_as_seg[:,:,aorta_mid].T, cmap='gray')
    # plt.title('after and')
    # plt.show()

    ROI_as_seg = np.subtract(ROI_as_seg, skin_segmentation)
    # plt.imshow(ROI_as_seg[:, :, aorta_mid].T, cmap='gray')
    # plt.title('after sub')
    # plt.show()

    # ROI_as_seg[ROI_as_seg != 1] = 0

    ct_data[:, :, :] = 0
    ct_data[ROI_as_seg == 1] = 1

    nib.save(ct_img, 'ROI_seg.nii.gz')
    print('ROI done')
    print('slice:', aorta_mid)

    # plt.imshow(ROI_as_seg.T, cmap='gray')
    # plt.title('ROI segmentation')
    # plt.show()

    return ROI_as_seg


def find_seeds(ct_scan, ROI):
    """
    A function that receives a CT scan and an ROI segmentation of the CT and returns a list of 200 seeds that are
    located within the liver
    """
    ct_img = nib.load(ct_scan)
    ct_data = ct_img.get_data()
    ROI_img = nib.load(ROI)
    ROI_data = ROI_img.get_data()

    seeds = []

    # find the borders of the ROI:
    upper_z = max([z if ROI_data[:, :, z].any() else 0 for z in range(ROI_data.shape[2])])
    lower_z = min([z if ROI_data[:, :, z].any() else 512 for z in range(ROI_data.shape[2])])
    upper_x = max([row if ROI_data[:, row, :].any() else 0 for row in range(ROI_data.shape[1])])
    lower_x = min([row if ROI_data[:, row, :].any() else 512 for row in range(ROI_data.shape[1])])
    left_y = max([col if ROI_data[col, :, :].any() else 0 for col in range(ROI_data.shape[0])])
    right_y = min([col if ROI_data[col, :, :].any() else 512 for col in range(ROI_data.shape[0])])

    # print('lower z:', lower_z)
    # print('upper z:', upper_z)
    # print('left y:', left_y)
    # print('right y:', right_y)
    # print('lower x:', lower_x)
    # print('upper x:', upper_x)

    # randomly sample points in the ROI and validate that they are in the liver
    while len(seeds) < SEEDS_NUM:
        # if not len(seeds) % 20:
            # print(len(seeds))

        z = np.random.randint(lower_z, upper_z + 1)
        x = np.random.randint(lower_x, upper_x + 1)
        y = np.random.randint(right_y + 30, left_y + 1)

        if LIVER_MIN_TH < ct_data[y, x, z] < LIVER_MAX_TH and ROI_data[y,x,z]:
            seeds.append([y, x, z])


    # ENABLE to see seeds as segmentation
    # ct_data[::] = 0
    # for seed in seeds:
    #     ct_data[seed[0], seed[1], seed[2]] = 1
    #
    # nib.save(ct_img, 'seeds_seg.nii.gz')

    return seeds





if __name__ == '__main__':
    # CT_scan = 'data/Case1_CT.nii.gz'
    # Aorta_seg = 'data/Case1_Aorta.nii.gz'
    # find_ROI(CT_scan, Aorta_seg)
    # ROI = 'ROI_seg.nii.gz'
    #
    # find_seeds(CT_scan, ROI)

    # segmentLiver('data/Case1_CT.nii.gz', 'data/Case1_Aorta.nii.gz', 'output')
    # liver_scan = nib.load('data/Case1_liver_segmentation.nii.gz')
    # liver_data = liver_scan.get_data()
    # print(np.sum(liver_data))
    #
    # CT_scan = nib.load('data/HardCase1_CT.nii.gz')
    # print(nib.aff2axcodes(CT_scan.affine))

    my_seg = nib.load('region_3D.nii.gz')
    seg_data = my_seg.get_data()

    for slc in range(my_seg.shape[2]):
        if seg_data[:,:,slc].any():
            seg_data[:,:,slc] = morphology.remove_small_holes(seg_data[:,:,slc].astype(np.uint8), area_threshold=200)
            seg_data[:,:,slc], connected_components_num = measure.label(seg_data[:,:,slc], return_num=True)
            props = measure.regionprops(seg_data[:,:,slc].astype(np.uint16), coordinates='rc')
            area_list = [region.area for region in props]
            seg_data[:,:,slc] = morphology.remove_small_objects(seg_data[:,:,slc].astype(np.uint8), min_size=max(area_list))

    # seg_flags = morphology.remove_small_holes(seg_data.astype(np.uint8), 200)
    # seg_data[:,:,:] = 0
    # seg_data[seg_flags] = 1

    # for slc in range(my_seg.shape[2]):
    # seg_data[:, :,:], new_connected_components_num = measure.label(seg_data, return_num=True)
    # print('before objects')

    # seg_data[:, :,:] = morphology.remove_small_objects(seg_data.astype(np.uint16))
        # seg_flags = morphology.remove_small_objects(seg_data.astype(np.uint8))
        # seg_data[:, :, :] = 0
        # seg_data[seg_flags] = 1

    seg_data[seg_data != 0] = 1

    nib.save(my_seg, 'seg_morphology.nii.gz')