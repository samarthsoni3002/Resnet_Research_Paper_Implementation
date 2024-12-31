import torch
import torch.nn as nn 
from torch.utils.data import Dataset 
from utils import get_class_names
from torchvision import transforms
from PIL import Image



class CreateDataset(Dataset):
    
    def __init__(self,
                 image_directory: str,
                 data_transform: transforms = None):
        
        super().__init__()
        
        self.paths =  list(image_directory.glob("*/*.jpg"))
        self.transform  = data_transform
        self.classes, self.class_to_idx, self.idx_to_class = get_class_names(images_directory=image_directory)
        
        
        
    def __len__(self):
        return len(self.paths)
    
    def load_images(self,index):
        image_path = self.paths[index]
        img = Image.open(image_path)
        return img
    
    
    def __getitem__(self,index):
        image = self.load_images(index)
        class_names = self.paths[index].parent.name
        filtered_names = class_names.split("-")[1]
        class_to_idx = self.class_to_idx[filtered_names]
        
        if self.transform:
            transformed = self.transform(image)
            return transformed, class_to_idx
            
        else:
            return image,class_to_idx
        


