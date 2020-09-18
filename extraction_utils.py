import os
import numpy as np
from skimage.util import view_as_windows
from skimage import io
from PIL import Image
import torchvision.transforms.functional as F


def delete_prev_images(dir_name):
    """
    Deletes all the file in a directory.
    :param dir_name: Directory name
    """
    for the_file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def check_and_reshape(image, mask):
    """
    Gets an image reshapes it and returns it with its mask.
    :param image: The image
    :param mask: The mask of the image
    :returns: the image and its mask
    """
    if image.shape == mask.shape:
        return image, mask
    elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
        return image, mask
    else:
        return image, mask


def extract_all_patches(image, window_shape, stride, num_of_patches, rotations, output_path, im_name, rep_num, mode):
    """
    Extracts all the patches from an image.
    :param image: The image
    :param window_shape: The shape of the window (for example (128,128,3) in the CASIA2 dataset)
    :param stride: The stride of the patch extraction
    :param num_of_patches: The amount of patches to be extracted per image
    :param rotations: The amount of rotations divided equally in 360 degrees
    :param output_path: The output path where the patches will be saved
    :param im_name: The name of the image
    :param rep_num: The amount of repetitions
    :param mode: If we account rotations 'rot' or nor 'no_rot'
    """
    non_doctored_windows = view_as_windows(image, window_shape, step=stride)
    non_doctored_patches = []
    for m in range(non_doctored_windows.shape[0]):
        for n in range(non_doctored_windows.shape[1]):
            non_doctored_patches += [non_doctored_windows[m][n][0]]
    # select random some patches, rotate and save them
    save_patches(non_doctored_patches, num_of_patches, mode, rotations, output_path, im_name, rep_num,
                 patch_type='authentic')


def save_patches(patches, num_of_patches, mode, rotations, output_path, im_name, rep_num, patch_type):
    """
    Saves all the extracted patches to the output path.
    :param patches: The extracted patches
    :param num_of_patches: The amount of patches to be extracted per image
    :param mode: If we account rotations 'rot' or nor 'no_rot'
    :param rotations: The amount of rotations divided equally in 360 degrees
    :param output_path: The output path where the patches will be saved
    :param im_name: The name of the image
    :param rep_num: The amount of repetitions
    :param patch_type: The mask of the image
    """
    inds = np.random.choice(len(patches), num_of_patches, replace=False)
    if mode == 'rot':
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'doctored' else patches[ind]
            for angle in rotations:
                im_rt = F.rotate(Image.fromarray(np.uint8(image)), angle=angle, resample=Image.BILINEAR)
                im_rt.save(output_path + '/{0}/{1}_{2}_{3}_{4}.jpg'.format(patch_type, im_name, i, angle, rep_num))
    else:
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'doctored' else patches[ind]
            io.imsave(output_path + '/{0}/{1}_{2}_{3}.jpg'.format(patch_type, im_name, i, rep_num), image)


def create_dirs(output_path):
    """
    Creates the directories to the output path.
    :param output_path: The output path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + '/authentic')
        os.makedirs(output_path + '/doctored')
    else:
        if os.path.exists(output_path + '/authentic'):
            delete_prev_images(output_path + '/authentic')
        else:
            os.makedirs(output_path + '/authentic')
        if os.path.exists(output_path + '/doctored'):
            delete_prev_images(output_path + '/doctored')
        else:
            os.makedirs(output_path + '/doctored')


def find_doctored_patches(image, im_name, mask, window_shape, stride, patches_per_image):
    """
    Gets an image reshapes it and returns it with its mask.
    :param image: The image
    :param im_name: The name of the image
    :param mask: The mask of the image
    :param window_shape: The shape of the window (for example (128,128,3) in the CASIA2 dataset)
    :param stride: The stride of the patch extraction
    :param dataset: The name of the dataset
    :param patches_per_image: The amount of patches to be extracted per image
    :returns: the doctored patches and their amount
    """
    # extract patches from images and masks
    patches = view_as_windows(image, window_shape, step=stride)
    mask_patches = view_as_windows(mask, window_shape, step=stride)

    doctored_patches = []
    # find doctored patches
    for m in range(patches.shape[0]):
        for n in range(patches.shape[1]):
            im = patches[m][n][0]
            ma = mask_patches[m][n][0]
            num_zeros = (ma == 0).sum()
            num_ones = (ma == 255).sum()
            total = num_ones + num_zeros
            if num_ones <= 0.99 * total:
                doctored_patches += [(im, ma)]

    # if patches are less than the given number then take the minimum possible
    num_of_patches = patches_per_image
    if len(doctored_patches) < num_of_patches:
        print("Number of doctored patches for image {} is only {}".format(im_name, len(doctored_patches)))
        num_of_patches = len(doctored_patches)

    return doctored_patches, num_of_patches
