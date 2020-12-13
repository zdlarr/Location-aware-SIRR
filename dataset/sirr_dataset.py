import torch
import os
from dataset.image_folder import make_dataset, default_loader
from PIL import Image
from utils.util import make_power
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

class DatasetDataLoader(object):
    
    def __init__(self, opts):
        super(DatasetDataLoader, self).__init__()
        self.opts = opts
        self.dataset = SIRRDataset(opts)
        print('Dataset [%s] was created' % type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=opts.batch_size,
            shuffle=not opts.serial_batches,
            num_workers=int(opts.num_threads)
        )

    def get_length(self):
        return len(self.dataset)
    
    def __len__(self):
        # return the number of dataset.
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data
        """
        for i, data in enumerate(self.dataloader):
            yield data


class SIRRDataset(data.Dataset):

    def __init__(self, opts):
        self.opts = opts
        # data root 
        self.data_root = opts.data_root
        self.I_dir = os.path.join(self.data_root, 'blend')

        self.I_paths = sorted(make_dataset(self.I_dir))
        self.Is_size = len(self.I_paths)

        # tensor utils.
        self.to_tensor = transforms.ToTensor()
    
    def __getitem__(self, index):
        I_path = self.I_paths[index]
        m_img = Image.open(I_path).convert('RGB')
        
        # Due the recurrent structure of our model and the min resolution of feature is origin's 1/8. 
        m_img = make_power(m_img, base=8)

        m = self.to_tensor(m_img)
        return {
            'I': m,
            'I_path': I_path
        }
    
    def __len__(self):
        return self.Is_size

    
