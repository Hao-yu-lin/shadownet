import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def has_files(fname, extensions):
    filename_lower = fname.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(root, dir, extensions):
    images = []
    fnames = []
    
    dir = os.path.join(root, dir)
    
    for names in sorted(os.listdir(dir)):
        for ext in extensions:
            if(names.endswith(ext)):
                path = os.path.join(dir, names)
                images.append(path)
                names = names.split(".")[0]
                fnames.append(names)
                break
    
    return images, fnames
    

class DatasetFolder(Dataset):
    def __init__(self, root, extensions, shadow_state, transform=None):
        
        img, fnames= make_dataset(root, shadow_state[0], extensions)
        
        self.root = root
        self.extensions = extensions
        self.samples = img
        self.fnames = fnames
        
        if(transform is None):
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            
            ])
        else:
            self.transform = transform
            
        self.intr2d_tfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ])
        
        self.intr2d_path = os.path.join(root, shadow_state[1])
        print(" ------ data : ", len(self.samples), " ------- ")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path)
        sample = self.transform(sample)
        
        intr2d_path = os.path.join(self.intr2d_path, self.fnames[index])
        intr2d_path = intr2d_path + ".png"
        intr2d = Image.open(intr2d_path)
        # print(intr2d)
        intr2d = self.intr2d_tfm(intr2d)
        
        # print(intr2d)
        return sample, intr2d


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImageFolder(DatasetFolder):
    def __init__(self, root, shadow_state, transform=None):
        super(ImageFolder, self).__init__(root, IMG_EXTENSIONS, shadow_state, transform)
    
        self.imgs = self.samples