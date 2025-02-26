
'''
Constains functionallity for creating PyTorch DataLoader's for
image data.
'''


import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.measure import find_contours
from scipy.fft import fft
import albumentations as A
import torchvision
from utils import get_fourier

from PIL import Image


class Custom_dataset(Dataset) :
  ''' Custom dataset for images and instance masks to perform CPN'''
  def __init__(self , image_dir , mask_dir , image_transform: torchvision.transforms = None ,
    image_augmentation_transform : torchvision.transforms = None ,
    mask_transform : torchvision.transforms = None ,
    d1_mask_transform : torchvision.transforms = None ,
    d2_mask_transform : torchvision.transforms = None ,
    num_coeffs: int = 20 , num_instances: int = 600 , batch_size: int = 2 ,
    shape2 : int = 512 , shape1 : int = 1024) :   # other pssible shape would be p2 256 , p1 512

    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.images = sorted(os.listdir(image_dir) )
    self.masks = sorted(os.listdir(mask_dir) )

    self.image_transform = image_transform   # Normalization of images

    self.mask_transform = mask_transform
    self.d1_mask_transform = d1_mask_transform
    self.d2_mask_transform = d2_mask_transform

    #shapes (shape , shape) for resize in coarse and reafine part:
    self.shape2 = shape2
    self.shape1 = shape1

    self.num_coeffs = num_coeffs
    self.num_instances = num_instances

  def __len__(self) :
    return len(self.images)

  def __getitem__(self , idx ) :
    image = np.array(Image.open(os.path.join(self.image_dir, self.images[idx])) )
    mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx]))    )

    #Change both types into float:
    image = image.astype('float32') / 255.0    # uint8
    mask = mask.astype('float32') / 65535.0   # uint16


    # #CHange one CC into 3 required CC, by resnet bb which represent grey image;
    image = torch.tensor(image)
    image = torch.stack([image, image, image], dim=0)    # Convert [1, H, W] -> [3, H, W]

    # Perform transformations:
    addi_transform = transforms.Compose([transforms.Resize( (1024,1024) )   ])
    mask = addi_transform(torch.tensor(mask) )
    image = addi_transform(image )

    if self.image_transform :
      image = self.image_transform(image)

    #For data augmentation:
    if self.mask_transform :

      # Apply same augmentation for image and mask based on albumentations library, which operates on opencv and numpy:

      #make necessary changes:
      mask = mask.unsqueeze(0).permute(1,2,0)
      image = image.permute(1,2,0)

      augmented = self.mask_transform(image=np.asarray(image), mask=np.asarray(mask) , )

      # Get the augmented image and mask and move them back to correct tensor representations:
      image = torch.tensor( augmented["image"] )
      mask = torch.tensor((augmented["mask"]) )

      mask = mask.permute(2, 0 ,1).squeeze(0)
      image = image.permute(2,0,1)


    ####     D2 :       ####
    d2_mask =  mask.clone().detach()

    H = W = self.shape2

    d2_mask = torch.nn.functional.interpolate( mask.unsqueeze(0).unsqueeze(0) , size=( H , W ), mode='nearest-exact' )
    d2_mask = d2_mask.squeeze(0)

    if self.d2_mask_transform :
      d2_mask = self.d2_mask_transform(d2_mask)

    d2_mask = d2_mask.squeeze(0)

    d2_mask_bin = (d2_mask > 0).float()   # get binary mask for smallere spatial resolution


    ####     D1 :       ####

    if mask[0].shape == self.shape1 :
      d1_mask = mask.clone().detach()

    else :
      d1_mask = mask.clone().detach()

      H = W = self.shape1
      d1_mask = torch.nn.functional.interpolate( mask.unsqueeze(0).unsqueeze(0) , size=( H , W ), mode='nearest-exact' )
      d1_mask = d1_mask.squeeze(0)

    if self.d1_mask_transform :
      d1_mask = self.d1_mask_transform(d1_mask)

    d1_mask = d1_mask.squeeze(0)

    ####    COEFFS:  #####


    d2_coeffs = get_fourier(d2_mask.unsqueeze(0) , max_num_coefs= self.num_coeffs , max_instances = self.num_instances )

    # retrurns original image, and mas: mask in smaller spatial resolution , and its coefficients , madk in higher spatial resolution, after transforms.
    return image , mask  , d2_mask , d2_coeffs.squeeze(0)  , d1_mask

# --- Transforms ---
transform = transforms.Compose([
    transforms.RandomApply(
        [transforms.Lambda(lambda img: torchvision.transforms.functional.adjust_gamma(img, gamma=random.uniform(0.2 , 0.5), gain=random.uniform( 0.8 , 1.05)))],p=1  ),
    transforms.Normalize(mean = [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225] ) ,
])



test_brightness_transform = transforms.Compose([
    transforms.RandomApply(
        [transforms.Lambda(lambda img: torchvision.transforms.functional.adjust_gamma(img, gamma=random.uniform(0.4 , 0.6), gain=random.uniform( 1.25 , 1.4)))],p=1  ),
    transforms.Normalize(mean = [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225] ) ,
])


# Define your augmentations pipeline using Albumentations:
augmentation_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=270, p=0.4),  # Rotate by a random angle within (-270 , 270) , with black pixels
]  , is_check_shapes= False)



    # d2_mask_transform = transforms.Compose([
    # ])


    # d1_mask_transform = transforms.Compose([
    # ])
