

import cv2
import torch
from torch import nn

import zipfile
import requests
from pathlib import Path
import os
from PIL import Image

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.ndimage

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def get_data_zip(source : str ,
                  destination : str , ) -> Path :
  '''Get a zipdataset from source and unzips to destination.
  Returns data path.
  '''
  # Setup path to data folder
  data_path = Path('')
  image_path = data_path / destination

  #Make directory
  image_path.mkdir(parents = True , exist_ok = True )

  # Unzip target file :
  with zipfile.ZipFile(source , 'r' ) as zip_ref :
    print(f'Unzipping {source} data...')
    zip_ref.extractall(image_path)

  return image_path

def walk_through_dir(dir_path) :
  '''
  Walks through dir_path returning its contents.
  '''
  for dirpath , dirnames , filenames in os.walk(dir_path) :
    print(f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')
    print('----'*5)

# Set seeds, helper functon to try reproducible things for comparison
def set_seeds(seed: int = 39) :
  '''Sets random sets for torch operations.

  Args:
      seed (int , optional): Random seed to set. Defaults to 808
  '''
  # Set the seed for general torch operations
  torch.manual_seed(seed)
  #Set the seed for CUDA torch operations, (ones that happen on the GPU)
  torch.cuda.manual_seed(seed)

  print(f'Seed set to: {seed}')

def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth'):
    print('Saving checkpoint\n')
    torch.save({
        'epoch': epoch + 1,              #########################
        'model_state_dict': model.state_dict() ,
        'optimizer_state_dict': optimizer.state_dict() ,    }, filename)


def load_checkpoint(model, optimizer, filename='checkpoint.pth' ):
  if os.path.isfile(filename):
    print('Loading a checkpoint\n')

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']
  else:
    print('Checkpoint file not found. Starting from scratch.\n')

    return 0


def save_model(model: torch.nn.Module,
               target_path: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model (torch.nn.Module): Target PyTorch model to save.
    target_dir (str): Directory for saving the model to.
    model_name (str): Filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="thats_nice_one_model.pth")
  """
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

  # If output_path not exist yet, create one:
  if not os.path.exists(target_path):
    os.makedirs(target_path , exist_ok = True )

  # Create target directory
  model_save_path = os.path.join(target_path , model_name)

  # Save the model state_dict()
  print(f"-> Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)



def create_writer(experiment_name : str ,
                  model_name : str ,
                  extra : str = None ) :

  '''Creates torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specyfic directory'''
  #One experiment one directory

  # Get timestamp of current date in reverse order:
  timestamp = datetime.now().strftime('%Y-%m-%d')

  if extra :
    #Create log directory path:
    log_dir = os.path.join('runs' , timestamp , experiment_name , model_name , extra)
  else :
    log_dir = os.path.join('runs' , timestamp , experiment_name , model_name)

  print(f'Created SummaryWriter saving to {log_dir}\n')
  return SummaryWriter(log_dir = log_dir)


#### EVALUATION ######

def save_loss_curves(results : Dict[str , List[float] ] , plot_path:str = 'loss_curve.png' ) :
  '''Plots training curves of a results dictionary.
  Args:
    results (Dict[str , List[float]): Dictionary of results e.g.
    train/test of loss value.
    plot_path (str): Path to save image. By default 'loss_curve.png'
  '''

  # Ensure the directory exists:
  plot_dir = os.path.dirname(plot_path)
  if not os.path.exists(plot_dir) and plot_dir != '':
      os.makedirs(plot_dir)

  # Get the loss values of the results dictionary (training and test)
  loss = results['train_loss']
  test_loss = results['test_loss']
  train_no_refine_loss = results['train_loss_no_ref']

  # Figure out how many epochs there were :
  epochs = range(len(results['train_loss']) )


  # Setup a plot:
  plt.figure(figsize = (15 , 7) )

  # Plotting training and validation loss
  if 'train_loss' in results and 'test_loss' in results:
    # plt.subplot(1 ,2 , 1)
    plt.plot(epochs, loss , label = 'train_loss')
    plt.plot(epochs , train_no_refine_loss , label = 'train_loss_no_ref')
    plt.plot(epochs, test_loss , label =  'test_loss')

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()


  # Save the plot
  plt.savefig(plot_path)
  plt.close()  # Close the figure to free up memory




def overlay_images(A, B):
    """Simple function to overlay image B into image A whenever B is not zero.
    Supports grayscale and color images of the same spatial size.

    Parameters:
    A (numpy.ndarray): Background image (HxW or HxWxC).
    B (numpy.ndarray): Overlay image (HxW or HxWxC).

    Returns:
    numpy.ndarray: Combined image.
    """
    # Ensure both images have the same shape
    if A.shape[:2] != B.shape[:2] :
      raise ValueError("Both images must have the same height and width")

    # Expand dimensions if one is grayscale and the other is color
    if A.ndim == 2:  # A is grayscale, make it CC
      A = np.stack([A, A , A] , axis = 2)

    if B.ndim == 2:  # B is grayscale, make it C
      B = np.stack([B , B , B] , axis = 2)

    # Create mask where B is not zero
    mask = np.any(B != 0, axis=-1, keepdims=True).astype(np.float32)

    # Apply mask to replace values in A with those in B
    C = np.where(mask, B, A)

    return C


def downsample_points(contour , N , max_num_points : int ):
    """Downsample the given contour in form of coefficients up to
    a specified maximal number of points. Used inside other functions.

    Args:
      contour (torch.Tensor): Complex tensor of shape (N,).
      max_num_points (int): Desired maximal number of points in the output contour.
    Returns:
      torch.Tensor: Downsampled contour.
    """

    # If contour is already small enough, return it just unchanged:
    if contour.shape[0]   <= max_num_points:
      return contour

    while contour.shape[0] > max_num_points :
      contour = contour[::N]  # Takes every N-th point

      if contour.shape[0]   <= max_num_points:
        # # Pad with zeros if needed to make the number of coefficients consistent
        padding_size =  max_num_points - contour.shape[0]
        contour = torch.cat((contour, torch.zeros(padding_size, dtype=torch.complex64 )), dim=0)
        # print(f'Additional print 2 i return: {contour}')

        return contour


def create_instance_map(binary_map_batch, max_num_instances: int = 600):
    """Generate instance IDs using connected components method for a batch of binary classification maps, limiting the number of instances to `max_num_instances`.
    Args:
    - binary_map_batch (torch.Tensor): Batch of binary classification maps, shape (B, H, W).
    - max_num_instances (int): Maximum number of instances to keep for each image.

    Returns:
    - instance_map_batch (torch.Tensor): Batch of instance ID maps, shape (B, H, W).
    """
    # Get the device of the input tensor
    device = binary_map_batch.device
          # print(f'Device of binary map: {binary_map_batch.device}')


    # Initialize array to store the instance map and number of instances
          # print('Input binary map batch form shape:' ,binary_map_batch.shape)

    B, H, W = binary_map_batch.shape[0] , binary_map_batch.shape[1]  , binary_map_batch.shape[2]     #.shape
    instance_map_batch = torch.zeros_like(binary_map_batch  )   #  .long
    num_instances_batch = torch.zeros(B, dtype=torch.int32)

    # Convert the batch of binary maps to numpy (for connected component labeling)
    binary_map_batch_np = binary_map_batch.cpu().numpy().astype(int)

    # print(f'binary_map_batch_np device currently usedd: {binary_map_batch_np.device}')

    # Label connected components for the entire batch (can be done in one step)
    labeled_maps = np.zeros_like(binary_map_batch_np, dtype=np.float32)
    for i in range(B):
        labeled_maps[i], num_instances_batch[i] = scipy.ndimage.label(binary_map_batch_np[i] , output=np.float32)


    # Calculate the component sizes for each image
    # Vectorized component size calculation across the batch
    component_sizes = []
    for i in range(B):
        sizes = scipy.ndimage.sum(binary_map_batch_np[i], labeled_maps[i], index=np.arange(1, num_instances_batch[i] + 1))
        component_sizes.append(sizes)

    # Now, we will process each image in the batch
    for i in range(B):
        # Sort components by size in descending order and get the top `max_num_instances`
        sorted_indices = np.argsort(component_sizes[i])[::-1]  # Sort indices by size in descending order

        #NEW: take at most max num instances
            # print(f'Length of sorted indices: {len(sorted_indices)}')
        if len(sorted_indices) < max_num_instances :
          top_indices = sorted_indices
        else :
          top_indices = sorted_indices[:max_num_instances]  # Select top `max_num_instances`

        # Create a new labeled map with the top instances
        new_labeled_map = np.zeros_like(labeled_maps[i], dtype=np.float32)
        new_instance_id = 1
        for idx in top_indices:
            new_labeled_map[labeled_maps[i] == (idx + 1)] = new_instance_id
            new_instance_id += 1

        # Assign the new labeled map to the batch tensor
        instance_map_batch[i] = torch.tensor(new_labeled_map, device=device, dtype=torch.float) #.long

    instance_map_batch = instance_map_batch.to(device)

    return instance_map_batch

def get_fourier( mask , max_num_coefs :int = 64 , max_instances : int = 600 , min_boundary_points :int =  2) :
  '''Convert pixel instance mask into Fourier coefficients tensor representation with instance preservation.

  Args:
    mask (torch.Tensor): Instance mask shape: (BS, H, W)
    max_num_coefs (int) : maximum number of coefficients per instance.
    max_instances (int) : maximum number of instances
    min_boundary_points (int) : Number of points, below which instance is treated as empty: 0. By default 2 points.
  Returns:
    fourier_coeffs (torch.Tensor): Fourier coefficients image representation of shape (BS , num_instances, num_coefficients).   '''

  BS, H, W = mask.shape  # CC are not important here
  all_fourier_coeffs = []


  # Create binary mask: Set all non-zero values to 255
  binary_masks = torch.where(mask != 0, 255, 0).to(torch.uint8)   # I dont multiply it to 255 becouse all what different than 0 is 255

  # Iterate over each image in the batch and get unique instance IDs per img (excluding background 0)
  for b in range(BS)  :


    #Find ALL contours:
    contours, _ = cv2.findContours(np.asarray(binary_masks[b].cpu() ), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Prepare placeholder tensor to hold (contours) to Fourier coefficients for each instance in the batch:
    fourier_contours = []

      # for instance in tqdm(range(len(instance_ids) -1  ) ) :
    for instance in range(len(contours)   )  :

      contour = torch.tensor(contours[instance] , dtype = torch.complex64)  # 0 Couse there is only one contour, so no worries

      # Include or exclude small instances based on min_boundary_points
      if len(contour) < min_boundary_points:
        # Generate zeroed Fourier coefficients for small instances
        contour = torch.zeros(max_num_coefs, dtype=torch.complex64 , device =  mask.device)   # Dtype comple64 or complex128 think about it


      else:

        contour = contour.squeeze(axis=1)  # Removes the middle dimension
        contour = contour[:, 0] + 1j * contour[:, 1]


      # Retain only the first max_num_coefs, all objects must have less than max_num_coefs of contour points !
      # selected_coeffs = contour[:max_num_coefs]

      selected_coeffs = downsample_points(contour= contour , N = 2 , max_num_points = max_num_coefs) # if it greater than max_num_points returns same number of points as input


      # Pad with zeros if needed to make the number of coefficients consistent
      if selected_coeffs.shape[0] < max_num_coefs:

        padding_size =  max_num_coefs - selected_coeffs.shape[0]
        selected_coeffs = torch.cat((selected_coeffs.to(device = mask.device) , torch.zeros(padding_size, dtype=torch.complex64 ,device =  mask.device)), dim=0)


      ### -----------------------------------------------------

      # Compute FFT
      fft_coeffs = torch.fft.fft(selected_coeffs , ).to( device =  mask.device)

      # Append the computed coefficients
      fourier_contours.append(fft_coeffs)


    # Pad the fourier_coeffs to have the same number of instances
    while len(fourier_contours) < max_instances:
      # Pad the Fourier coefficients for missing instances with zero coefficients
      fourier_contours.append(torch.zeros(max_num_coefs, dtype=torch.complex64 , device =  mask.device  )) #

    # Stack the Fourier coefficients to create a tensor of shape (max_instances, max_num_coefs)
    fourier_contours = torch.stack(fourier_contours, dim=0 )

    # Add this image's Fourier coefficients to the batch list
    all_fourier_coeffs.append(fourier_contours )

    # print(f'Number of valid Fourier coefficients for image ALL : {(torch.stack(all_fourier_coeffs, dim=0)).shape}')  # Debugging
    f_m = torch.stack(all_fourier_coeffs, dim=0).to(device =  mask.device)

  # Stack all images in the batch into a tensor of shape (B, max_instances, max_num_coefs)
  return f_m

def make_pixel(coefficients , img_shape,  wdyw: str = 'area') :
  '''Convert coefficient fourier representation into pixel space contour representation.
  Args:
    coefficients (torch.Tensor): Tensor form representation of image: [BS, num_instances , num_coefficients]
    img_shape (tuple): Shape of original img, to regain.
    wdyw (str): Define do you wonna result in form of area or contours. By defult set to 'area'. Possible options: ['area' , 'contours']

  Returns:
    empty_mask , contours (tuple): Binary map of predicted instances all on one img, contour tenros represetnation: [BS , num_instance, num_points , 2]
  '''

  BS , num_instance , num_coefs = coefficients.shape

  H , W = img_shape

  empty_mask = np.zeros(shape = (H, W) , dtype = np.uint8)

  #Draw contours on a copy of the original image
  empty_mask = cv2.cvtColor(empty_mask, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

  all_contours = []

  img_stack = []

  for b in range(BS)  :
    img = coefficients[b]

    fourier_contours = []
    for instance in range(num_instance)   :

      contour = img[instance]

      reconstructed_complex = torch.fft.ifft( contour )


      # Convert back to integer coordinates
      reconstructed_contours = torch.column_stack((reconstructed_complex.real, reconstructed_complex.imag))

      reconstructed_contours = torch.round(reconstructed_contours ).to(torch.int32)

      fourier_contours.append(reconstructed_contours)

      # Keep rows where at least one element is nonzero element:
      reconstructed_contours = reconstructed_contours[torch.any(reconstructed_contours != 0, dim=1)]


      if reconstructed_contours.size(0) == 0 :
        # print('\nEmpty tensor --> skip\n')
        continue


      #TO draw indyvidual contour:
      reconstructed_contours = reconstructed_contours.cpu().numpy().reshape(-1, 1, 2)  # OpenCV expects shape (N, 1, 2)

      if wdyw == 'area' :
        cv2.drawContours(empty_mask, reconstructed_contours , -1, (255, 255, 255), -1)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
      elif wdyw == 'contours' :
        cv2.drawContours(empty_mask, reconstructed_contours , -1, (255, 255, 255), 2)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
      else :
        raise ValueError("Value for wdyw may be choose from (str) 'area' or 'contours' only, to plot area or contour correctly.")


    fourier_contours = torch.stack(fourier_contours, dim=0)

    all_contours.append(fourier_contours )

    img_stack.append(empty_mask)


    f_m = torch.stack(all_contours, dim=0).to(device = coefficients.device )

  empty_mask = torch.stack([torch.from_numpy(empty_mask) for mask in img_stack])  # (BS, H , W, 3)

  # Stack all images in the batch into a tensor of shape (B, max_instances, max_num_coefs)

  return empty_mask , f_m

def reg_loss(gt_coefs, pred_coefs , device) :
  '''Allow for measuring regression loss for fourier coefficient representation.
  '''
  #FOREGROUND MASK FOR COEFS:

  #Create the foreground mask for each image in the batch:
  # For each image, check where the instances are non-zero
  foreground_mask = (gt_coefs.abs().sum(dim=-1) > 0).float()

  pred_coefs , gt_coefs = pred_coefs.to(device) , gt_coefs.to(device)

  loss_real = torch.nn.functional.mse_loss(pred_coefs.real, gt_coefs.real , )
  loss_imag = torch.nn.functional.mse_loss(pred_coefs.imag, gt_coefs.imag , )

  # Apply foreground mask (to ignore padded instances)
  masked_loss_real = loss_real * foreground_mask.unsqueeze(-1)  # Multiply by mask to ignore padded instances
  masked_loss_imag = loss_imag * foreground_mask.unsqueeze(-1)  # Multiply by mask to ignore padded instances

  mean_loss_real = masked_loss_real.sum(dim=1).sum(dim=1) / foreground_mask.sum(dim=1)  # Sum and normalize by number of valid instances
  mean_loss_imag =  masked_loss_imag.sum(dim=1).sum(dim=1) / foreground_mask.sum(dim=1)

  # Final step: Average the loss across the batch (average loss for all images in the batch)
  batch_loss_real = mean_loss_real.mean()  # Final loss for the batch
  batch_loss_imag = mean_loss_imag.mean()

  regression_loss = batch_loss_real + batch_loss_imag
  return regression_loss


###   REFINEMENT LOSS   ###

def refinement_loss(refined_contours, gt_contours):
    """Computes a scalable loss that penalizes further distances more and ignore padded (0,0) points.
    Args:
        refined_contours (Tensor): Predicted refined contour points (B, N, P, 2)
        gt_contours (Tensor): Ground truth contour points (B, N, P, 2)
    Returns:
        loss (Tensor): Scalable loss value
    """
    # Foreground mask: ignore (0,0) padded points
    foreground_mask = (gt_contours.abs().sum(dim=-1) > 0).float()  # Shape: (B, N, P)

    # Compute Euclidean distance per point
    distances = torch.norm(refined_contours.round() - gt_contours, dim=-1)  # Shape: (B, N, P): ROUNDED CONTOURS

    #Further points contribute more to the loss with square, loss ignores padded 0th points
    loss = (distances**2) * foreground_mask

    # Normalize by number of valid elements (to avoid bias)
    valid_count = foreground_mask.sum().clamp(min=1)  # Avoid division by zero !!!
    return loss.sum() / valid_count  # Mean loss over valid points


###   REFINE    ####


def refine(refined_contours , mask_contours , gt_contours = None, refinement_field = None , img_shape = (1024,1024) ,  sigma : float = 3.  ) :
  '''Function for refinment step, make a step correct with refinment_field, and return the loss.

  Args:
    refined_contours (torch.Tensor) : Contours of shape: [BS ,  num_instances , num_points , 2 ] which
    will be refined during step.
    mask_contours (torch.Tensor) : Contours of shape: [BS ,  num_instances , num_points , 2 ], used for
    masking purposes, to skip zeroes.
    gt_contours (torch.Tensor): tensor form contours of ground truth [BS , num_instances , num_points , 2].
    Used to loss calculation. By default set to None, if so skip loss calculation, for example testing.
    refinement_field (torch..Tensor): Refinement filed learned by refinment head.
    sigma (float) : Mulitplier of refinement correction during, one step.

  Return:
    loss (torch.Tensor) : loss to acumulate during each step in refinement loop.
    refined_contours (torch.Tensor) : Corrected refined_contours in this step; shape: [BS ,  num_instances , num_points , 2 ]
  '''

  #  Round and clamp coordinates
  int_coords = refined_contours.round().long()
  int_coords[..., 0] = int_coords[..., 0].clamp(0, img_shape[1] - 1)  # 512
  int_coords[..., 1] = int_coords[..., 1].clamp(0, img_shape[0] - 1)   # 512

  B, N, P, _ = refined_contours.shape
  batch_indices = torch.arange(B, device=int_coords.device).view(B, 1, 1).expand(B, N, P)

  #Extract dx_dy from refinement field
  dx_dy = refinement_field.permute(0, 2, 3, 1)  # (B, H, W, 2)
  dx_dy = dx_dy[batch_indices, int_coords[..., 1], int_coords[..., 0]]  # (B, N, P, 2)


  # Normalize dx_dy per instance to prevent collapsing
  dx_dy = dx_dy - dx_dy.mean(dim=2, keepdim=True)  # Center per instance


  #Scale corrections by sigma factor
  correction = sigma * torch.tanh(dx_dy)

  #Create a per-instance mask to prevent merging
  mask = (mask_contours != 0).any(dim=-1).unsqueeze(-1).expand(-1, -1, -1, 2)

  # Apply mask to avoid modifying non-contour points
  correction = correction * mask

  #Update contour coordinates
  refined_contours = refined_contours + correction


  if gt_contours != None :
    loss = refinement_loss(refined_contours = refined_contours , gt_contours = gt_contours)

    loss = torch.tensor([loss])
    return loss , torch.tensor(refined_contours , device = refined_contours.device)

  # IF gt is provided, then return just, refined contours, no loss calculation!
  return torch.tensor(refined_contours  , device = refined_contours.device)
import numpy as np
import cv2
import torch

def draw_refined_contours(contours, coarse = None , image_size=(512, 512) , wdyw = 'area' , color = True):
  '''Function allow to draw contours representation of [ BS , N , P ,2].
  contours (torch.Tensor) : contours to draw [ BS , N , P ,2].
  coarse (torch.Tensor) : By default set to None, otherwise input a background contours.
  wdyw (str): Define do you wonna result in form of area or contours. By defult set to 'area'. Possible options: ['area' , 'contours']
  color (bool) : If set to True draw purple 3 CC contours, otherwise draw all contours white.
  '''
  BS, n, points, _ = contours.shape  # Batch Size, number of instances, number of points, 2

  # Create a separate mask for each batch
  batch_masks = []

  if contours is not None :
    contours = torch.tensor(contours)

  for b in range(BS) :
    # Create a new mask for each batch instance
    empty_mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    empty_mask = cv2.cvtColor(empty_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR

    contours_b = contours[b]


    if coarse is not None :
      coarse_b = coarse[b]

      # loop through instances:
      for i in range(n) :

        # Keep rows where at least one element is nonzero:
        reconstructed_contours = coarse_b[i][ torch.any(contours_b[i] != 0, dim=-1) ]
        reconstructed_contours = ( reconstructed_contours.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.int32))

        # Skip empty contours
        if reconstructed_contours.shape[0] == 0:
            continue

        if color :
          if wdyw == 'area' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 0, 255), -1)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          elif wdyw == 'contours' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 0, 255), 2)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          else :
            raise ValueError("Value for wdyw may be choose from (str) 'area' or 'contours' only, to plot area or contour correctly.")

        else :
          if wdyw == 'area' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 255, 255), -1)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          elif wdyw == 'contours' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 255, 255), 2)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          else :
            raise ValueError("Value for wdyw may be choose from (str) 'area' or 'contours' only, to plot area or contour correctly.")

    for i in range(n) :

      # Keep rows where at least one element is nonzero:
      reconstructed_contours = contours_b[i][ torch.any(contours_b[i] != 0, dim=-1) ]
      reconstructed_contours = ( reconstructed_contours.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.int32))

      # Skip empty contours
      if reconstructed_contours.shape[0] == 0:
          continue

      if color :
          if wdyw == 'area' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 0, 255), -1)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          elif wdyw == 'contours' :
            cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 0, 255), 2)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
          else :
            raise ValueError("Value for wdyw may be choose from (str) 'area' or 'contours' only, to plot area or contour correctly.")
      else :
        if wdyw == 'area' :
          cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 255, 255), -1)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
        elif wdyw == 'contours' :
          cv2.drawContours(empty_mask, ([reconstructed_contours]) , -1, (255, 255, 255), 2)  # -1: draw all contours, you can draw one by one, Draw in purple , contour thickness
        else :
          raise ValueError("Value for wdyw may be choose from (str) 'area' or 'contours' only, to plot area or contour correctly.")


    batch_masks.append(empty_mask )  # Save mask for this batch


  # res = torch.stack(batch_masks, dim=0)
  res = torch.stack([torch.from_numpy(empty_mask) for mask in batch_masks])  # (BS, 512, 512, 3)


  return res # Returns masks

