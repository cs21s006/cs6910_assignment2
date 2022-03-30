import cv2
import glob
import random
import numpy as np
import torch
from pandas.core.common import flatten
torch.manual_seed(7)
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch.utils.data import DataLoader,Dataset


class iNaturalist(Dataset):
    '''Custom dataset class for iNaturalist dataset.
    '''
    def __init__(self, image_paths, class_to_idx, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx= class_to_idx
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        
        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        PIL_image = self.transform(PIL_image)

        return PIL_image, label


def create_data(data_type, data_path,  data_aug, image_shape, b_size):
  #Defining transformations when data_aug=True  [used when data_type='train' and data_aug=True]
  true_transforms = T.Compose([T.Resize((image_shape)),
                              T.RandomRotation(degrees=30),
                              T.RandomHorizontalFlip(p=0.5),
                              T.RandomGrayscale(p=0.2),
                              T.ToTensor(),
                                        ])
  
  #Defining transformations when data_aug=False
  false_transforms = T.Compose([T.Resize((image_shape)),
                               T.ToTensor()
                                  ])

  image_paths=[] # List to stoe image paths
  classes= [] # List to  store class values

  #get all the paths from data_path and append image paths and class to to respective lists
  for curr_data_path in glob.glob(data_path + '/*'):
    classes.append(curr_data_path.split('/')[-1]) 
    image_paths.append(glob.glob(curr_data_path + '/*'))

  image_paths = list(flatten(image_paths))

  #Creating dictionary for class indexes
  idx_to_class = {i:j for i, j in enumerate(classes)}
  class_to_idx = {value:key for key,value in idx_to_class.items()}

  if data_type == 'test':
    test_image_paths=image_paths

    #Using custom class for getting test dataset
    test_dataset= iNaturalist(test_image_paths,class_to_idx,false_transforms)

    #using Dataloader to load test dataset according to batch size
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=True)

    return test_loader


  else:
    random.shuffle(image_paths)

    #Setting aside 10% of the training data as validation data for hyperparameter tuning
    train_image_paths, valid_image_paths = image_paths[:int(0.9*len(image_paths))], image_paths[int(0.9*len(image_paths)):] 

    #Using custom class for getting train and validation dataset
    if data_aug == True:
      train_dataset = iNaturalist(train_image_paths,class_to_idx,true_transforms)
      valid_dataset = iNaturalist(valid_image_paths,class_to_idx,false_transforms)  
    else:
      train_dataset = iNaturalist(train_image_paths,class_to_idx,false_transforms)
      valid_dataset = iNaturalist(valid_image_paths,class_to_idx,false_transforms)  


    #using Dataloader to load train and valid dataset according to batch size
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=b_size, shuffle=True)

    return train_loader,valid_loader