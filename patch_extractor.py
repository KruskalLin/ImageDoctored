from skimage import io
import os
from extraction_utils import check_and_reshape, extract_all_patches, create_dirs, save_patches, find_doctored_patches
from skimage.color import gray2rgb


class PatchExtractor:
    def __init__(self, input_path, output_path, patches_per_image=4, rotations=8, stride=8, mode='no_rot'):
        """
        Initialize class
        :param patches_per_image: Number of samples to extract for each image
        :param stride: Stride size to be used
        """
        self.patches_per_image = patches_per_image
        self.stride = stride
        rots = [0, 90, 180, 270]
        self.rotations = rots[:rotations]
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path
        self.window_shape = (128, 128, 3)

    def extract_authentic_patches(self, im_name, num_of_patches, rep_num):
        """
        Extracts and saves the patches from the authentic image
        :param sp_pic: Name of doctored image
        :param num_of_patches: Number of patches to be extracted
        :param rep_num: Number of repetitions being done(just for the patch name)
        """
        au_image = io.imread('./original_images/' + im_name + '.jpg')
        extract_all_patches(au_image, self.window_shape, self.stride, num_of_patches, self.rotations, self.output_path,
                            im_name, rep_num, self.mode)

    def extract_patches(self):
        """
        Main function which extracts all patches
        :return:
        """
        # create necessary directories
        create_dirs(self.output_path)

        # define window shape
        tp_dir = self.input_path + '/images/'
        # run for all the doctored images
        for rep_num, f in enumerate(os.listdir(tp_dir)):
            image = io.imread(tp_dir + f)
            im_name = f.split(os.sep)[-1].split('.')[0]
            # read mask
            mask = gray2rgb(io.imread(self.input_path + '/masks/' + im_name + '.jpg'))
            image, mask = check_and_reshape(image, mask)

            # extract patches from images and masks
            doctored_patches, num_of_patches = find_doctored_patches(image, im_name, mask,
                                                                     self.window_shape, self.stride,
                                                                     self.patches_per_image)
            save_patches(doctored_patches, num_of_patches, self.mode, self.rotations, self.output_path, im_name,
                         rep_num, patch_type='doctored')
            self.extract_authentic_patches(im_name, num_of_patches, rep_num)