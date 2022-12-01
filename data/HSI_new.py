import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils
import numpy as np
import torch
from scipy.io import loadmat


class HSI(data.Dataset):
    """

    Keyword arguments:
    - root_dir (``string``): Root directory path.
 os   - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = "train/"  # train data
    train_lbl_folder = "train/"  # train label

    # Validation dataset root folders
    val_folder = "val/"  # val data
    val_lbl_folder = "val/"  # val label

    # Test dataset root folders
    test_folder = "test/"
    test_lbl_folder = "test/"

    # Filters to find the images
    img_extension = '.png'
    lbl_extension = '.mat'
    lbl_name_filter = 'label'

    # The values associated with the 17 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    # indian pines
    color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('Alfalfa', (128, 64, 128)),
        ('Corn-notill', (244, 35, 232)),
        ('Corn-mintill', (70, 70, 70)),
        ('Corn', (102, 102, 156)),
        ('Grass-pasture', (190, 153, 153)),
        ('Grass-trees', (153, 153, 153)),
        ('Grass-pasture-mowed', (250, 170, 30)),
        ('Hay-windrowed', (220, 220, 0)),
        ('Oats', (107, 142, 35)),
        ('Soybean-nottill', (152, 251, 152)),
        ('Soybean-mintill', (70, 130, 180)),
        ('Soybean-clean', (220, 20, 60)),
        ('wheat', (255, 0, 0)),
        ('Woods', (0, 0, 142)),
        ('Buildings-Grass-Trees-Drives', (0, 0, 70)),
        ('Stone-Steel-Towers', (0, 60, 100))
    ])

    # paviaU
    # color_encoding = OrderedDict([
    #     ('unlabeled', (0, 0, 0)),
    #     ('Asphalt', (128, 64, 128)),
    #     ('Meadows', (244, 35, 232)),
    #     ('Gravel', (70, 70, 70)),
    #     ('Trees', (102, 102, 156)),
    #     ('Painted metal sheets', (190, 153, 153)),
    #     ('Bare Soil', (153, 153, 153)),
    #     ('Bitumen', (250, 170, 30)),
    #     ('Self-Blocking Bricks', (220, 220, 0)),
    #     ('Shadows', (107, 142, 35))
    # ])

    # salinas
    # color_encoding = OrderedDict([
    #     ('unlabeled', (0, 0, 0)),
    #     ('Brocoli_green_weeds_1', (0, 60, 100)),
    #     ('Brocoli_green_weeds_2', (128, 64, 128)),
    #     ('Fallow', (244, 35, 232)),
    #     ('Fallow_rough_plow', (70, 70, 70)),
    #     ('Fallow_smooth', (102, 102, 156)),
    #     ('Stubble', (190, 153, 153)),
    #     ('Celery', (153, 153, 153)),
    #     ('Grapes_untrained', (250, 170, 30)),
    #     ('Soil_vinyard_develop', (220, 220, 0)),
    #     ('Corn_senesced_green_weeds', (107, 142, 35)),
    #     ('Lettuce_romaine_4wk', (152, 251, 152)),
    #     ('Lettuce_romaine_5wk', (70, 130, 180)),
    #     ('Lettuce_romaine_6wk', (220, 20, 60)),
    #     ('Lettuce_romaine_7wk', (255, 0, 0)),
    #     ('Vinyard_untrained', (0, 0, 142)),
    #     ('Vinyard_vertical_trellis', (0, 0, 70))
    # ])

    # The values above are remapped to the following
    new_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    # Default encoding for pixel value, class name, and class color

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.loader = loader

        self.targets = []

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths list
            # print('====> root dir: ', self.root_dir)
            # print('====> train_folder: ', self.train_folder)
            # print(os.path.join(self.root_dir, self.train_folder))
            self.train_data = utils.get_files(
                os.path.join(self.root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_extension)
            # print(self.train_labels)

            tar = []
            for data_path in self.train_data:
                # print(data_path.split('/'))
                # print(data_path)
                label = data_path.split('/')[-2]
                # print(type(label), label)
                tar.append(int(label))
            for i in range(len(tar)//4):
                self.targets.append(tar[i*4])
            # print('====>targets:', len(self.targets))
            # print(self.targets)

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths list
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)
            # print(self.val_data)
            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path = [self.train_data[index*4], self.train_data[index*4+1], self.train_data[index*4+2], self.train_data[index*4+3]]
            label_path = self.root_dir + self.train_lbl_folder + self.train_data[index*4].split('/')[5] + '/label.mat'
            # print(data_path)
            # print(label_path)
        elif self.mode.lower() == 'val':
            data_path = [self.val_data[index*4], self.val_data[index*4+1], self.val_data[index*4+2], self.val_data[index*4+3]]
            label_path = self.root_dir + self.val_lbl_folder + self.val_data[index*4].split('/')[5] + '/label.mat'
        elif self.mode.lower() == 'test':
            data_path = [self.test_data[index*4], self.test_data[index*4+1], self.test_data[index*4+2], self.test_data[index*4+3]]
            label_path = self.root_dir + self.test_lbl_folder + self.test_data[index*4].split('/')[5] + '/label.mat'
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        [img0, img1, img2, img3], label = self.loader(data_path, label_path)
        # print(label)
        # print(np.argwhere(label == 1)[0][1])
        # self.targets.append(np.argwhere(label == 1)[0][1])
        # print('====>targets:', len(self.targets))

        # Remap class labels
        # label = utils.remap(label, self.full_classes, self.new_classes)

        if self.transform is not None:
            [img0_0, img0_1, img0_2, img0_3] = self.transform[0](img0, img1, img2, img3)
            [img1_0, img1_1, img1_2, img1_3] = self.transform[1](img0, img1, img2, img3)

        target = np.argwhere(label)
        target = target[0][1]
        # print('====>target:', type(target), target)
        # target = np.squeeze(np.transpose(label, (1, 0)))
        # target = torch.from_numpy(target.astype(np.float32))
        # print(img0.shape, label.shape)

        return [img0_0, img0_1, img0_2, img0_3], [img1_0, img1_1, img1_2, img1_3], target

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)//4
        elif self.mode.lower() == 'val':
            return len(self.val_data)//4
        elif self.mode.lower() == 'test':
            return len(self.test_data)//4
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
