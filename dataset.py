import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

class ADNI_Dataset(Dataset):

    def __init__(self, config, data_csv, data_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.config = config
        self.data_type = data_type
        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)

        if self.data_type == 'train':
            if self.config.tf == "all_tf":
                self.transforms.register(Flip(), probability=0.5)
                self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.config.input_size)/4)), probability=0.5)
                self.transforms.register(Crop(np.ceil(0.75*np.array(self.config.input_size)), "random", resize=True),
                                        probability=0.5)

            elif self.config.tf == "cutout":
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.config.input_size)/4)), probability=0.7)

            elif self.config.tf == "crop":
                self.transforms.register(Crop(np.ceil(0.75*np.array(self.config.input_size)), "random", resize=True), probability=1)
        
        self.data_dir = config.data
        self.data_csv = data_csv
        self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.data_csv['SubjectID'])]
        
    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):
        labels = self.data_csv[self.config.label_name].values[idx]
        SubjectID = self.data_csv['SubjectID'].values[idx]
        file_match = [file for file in self.files if SubjectID in file]
        path = os.path.join(self.data_dir, file_match[0])
        img = nib.load(os.path.join(path, 'brain_to_MNI_nonlin.nii.gz'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)
        img = resize(img, (self.config.input_size[1], self.config.input_size[2], self.config.input_size[3]), mode='constant')
        img = torch.from_numpy(img).float().view(self.config.input_size[0], self.config.input_size[1], self.config.input_size[2], self.config.input_size[3])
        img = img.numpy()
        
        np.random.seed()
        x = self.transforms(img)

        return (x, labels)

    def __len__(self):
        return len(self.data_csv)
