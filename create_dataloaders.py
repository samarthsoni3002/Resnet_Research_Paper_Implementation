from dataset_class import CreateDataset

import torch
from torch.utils.data import DataLoader,random_split
from torchvision import transforms




def create_dataloader(images_directory: str,
                      batch_size: int,
                      data_transforms: transforms,
                      num_workers: int):
    
    
    image_dataset = CreateDataset(images_directory,data_transforms)

    test_size = int(0.2*len(image_dataset))
    train_size = len(image_dataset) - test_size
    train_dataset,test_dataset = random_split(image_dataset,[train_size,test_size])
    
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    
    
    return train_dataloader,test_dataloader