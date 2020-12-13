"""
    Modified from the original image folder file.
    From https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.
    This case can load images from both current directory & sub-directory
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(file_name):
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float('inf')):
    # get all images path in dir
    images_paths = []
    assert os.path.isdir(dir), '{} is not a valid dir'.format(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                full_path = os.path.join(root, fname)
                images_paths.append(full_path)
    
    return images_paths[:min(max_dataset_size, len(images_paths))]

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    
    def __init__(self, root_dir, transform=None, return_paths=False,
                 loader=default_loader):
        super(ImageFolder, self).__init__()
        self.images_paths = make_dataset(root_dir)
        if len(self.images_paths) == 0:
            raise(RuntimeError("Found 0 images in: " + root_dir + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.images_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img,  path
        else:
            return img
    
    def __len__(self):
        return len(self.images_paths)

