import numpy as np
from PIL import Image
import glob

import torch
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.image_list = glob.glob(folder_path+'*')
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        single_image_path = self.image_list[index]
        im_as_im = Image.open(single_image_path)
        im_as_np = np.asarray(im_as_im)/255
        im_as_np = np.expand_dims(im_as_np, 0)
        im_as_ten = torch.from_numpy(im_as_np).float()
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len
