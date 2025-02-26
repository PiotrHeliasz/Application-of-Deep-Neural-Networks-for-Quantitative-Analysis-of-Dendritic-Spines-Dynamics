
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

from utils import save_model , save_checkpoint , load_checkpoint , create_writer
from utils import create_instance_map , get_fourier , make_pixel , draw_refined_contours , reg_loss

from tqdm.auto import tqdm
from typing import Dict , List , Tuple

from utils import create_writer , save_loss_curves



def train(model: torch.nn.Module ,
          device : torch.device, # current device
          train_dataloader : torch.utils.data.DataLoader ,
          test_dataloader : torch.utils.data.DataLoader ,
          optimizer : torch.optim.Optimizer ,
          writer : torch.utils.tensorboard.writer.SummaryWriter ,
          epochs : int = 5 ,
          load_model: bool = True  ,
          model_name: str = 'new_model.pth' ,
          checkpoint_path: str = 'checkpoint.pth' ,
          alpha : int = 100 ,
          weights_object : int = 10 ,
          PENALTY_WEIGHT : int =  1e9 ,
          PENALTY_THRESH : float = 0.21 ,
          QUNATITY_PENALTY_WEIGHT: float = 4. ,
          P1_SHAPE : tuple = (512, 512) ,
          MAX_NUM_COEFS : int = 64 ,
          MAX_NUM_INSTANCES : int = 600 ,
          MIN_BOUNDARY_POINTS : int = 2 ,
          sigma = 3 ,
          num_iter = 3  ,
          do_refine : bool = True ) -> Dict[str, List] :

  ''' Complete function for training.
  Args:
    model (torch.nn.Module): Model to train.
    train_dataloader (torch.utils.data.DataLoader): Training dataloader.
    test_dataloader (torch.utils.data.DataLoader): Validating/testing dataloader.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    writer (torch.utils.tensorboard.writer.SummaryWriter):
    SummaryWriter() instance tracking to a specyfic directory.

    epochs (int): Number of epochs during training.
    device (torch.device): Device used for training.
    load_model (bool): By default load_model = True, for every epoch,
    model state_dict, optimizer, and epoch will be saved.
    model_name (str): Filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension. By default,
    model_name = 'new_model.pth' is saved in: 'models/new_model.pth'
    checkpoint_path (str): Path to checkpoint which you want to load,
    to continue training a model.
    By default checkpoint_path = 'checkpoint.pth'.
    'checkpoint.pth' is the path where checkpoint will overwrite itself.
    alpha (int) Multiplier of classification loss contribution.
    weights_object: (int) Multiplier of foreground pixel value.
    PENALTY_WEIGHT :  (int) Multiplier of penalty for predicting too much foreground class.
    PENALTY_THRESH: (float) Penalty border, of maximal acceptable foreground class pixels percentage.
    QUNATITY_PENALTY_WEIGHT: Multiplier for penalty for predicting more or less instances than marked on image mask.
    P1_SHAPE: (tuple) Declared shape of D1 spatial resolution.
    MAX_NUM_COEFS: (int) Maximal operated number of coefficients describing instance on image.
    MAX_NUM_INSTANCE: (int) (int) Maximal operated number of instances on image.
    MIN_BOUNDARY_POINTS: (int) Minimal number of points for object.
    sigma: Multiplier of refinment step value
    num_iter: Number of refine iterations
    do_refine: (bool) do perform refine? By default set to True


  Returns:
    A dictionary of training and testing loss.
    Each metric has a value in a list for each epoch.
    In the form: {train_loss: [...],
                 'train_loss_no_ref' : [...] ,
                 'test_loss': [...], }
    For example if training for epochs=3:
             {train_loss: [2.0616, 1.0537 , 0.8543],
             'train_loss_no_ref' : [2.0312, 1.0011 , 0.7612],
              test_loss: [1.2641, 1.5706 , 1.001], }
  '''

  # Create empty results dictionary:
  results = {'train_loss' : [] ,
             'train_loss_no_ref' : [] ,
             'test_loss' : [] ,
             'test_acc' : [] }    # possibly add more metrics , locc is calcualted here only bese on dendritic spines class prediction

  checkpoint_path = f'{model_name}_{checkpoint_path}'

  start_epoch = 0
  if load_model :
    start_epoch = load_checkpoint(model , optimizer , filename = checkpoint_path )

  assert start_epoch < epochs , f'You cannot continue training from {start_epoch} epoch to {epochs} epoch. Start epoch should be smaller than epochs!'


  model.train()



  ####   ####    ####   #####    #####    ####

  for epoch in tqdm(range(start_epoch , epochs ) ) :   # start to end

    train_loss  = test_loss = train_loss_print  = test_acc =  0

    i = 0

    for img, mask  , d2_mask , d2_coefs  , d1_mask ,  in tqdm(train_dataloader) :


      # TO skip to testing loop:
      #############
      # break
      ##########
      #
       #
         #
     

      #Send data to correct devices:
      img , mask , d2_mask , d2_coefs , d1_mask ,  = img.to(device) , mask.to(device) , d2_mask.to(device) , d2_coefs.to(device) , d1_mask.to(device)

    
      if do_refine :
        # return classification.to(device), coefs_p2.to(device)  , loss , refined_contours  # --> [Bs , 1, H , W] QQQ2 final_mask
        classification_map, coarse_predictions, loss  , refined_contours = model(img , gt_d1_mask =  d1_mask  , device = device, p1_shape =  P1_SHAPE , sigma = sigma , num_iter = num_iter )  # For high res give mask instead of d1_mask

        # To calc number of instances used in lower resolution classifcation:
        _ , coarse_contours = make_pixel(coefficients = coarse_predictions , img_shape = (d2_mask.shape[1] , d2_mask.shape[2]) , wdyw = 'area' )
        coarse_contours = coarse_contours.to(device)


        # Get binary refined map:
        binary_pred = draw_refined_contours(contours = refined_contours, image_size= P1_SHAPE , wdyw = 'area' , color = False)


        # BS , H , W  3
        binary_pred = binary_pred.to(device)
        binary_pred = binary_pred.permute(0, 3, 1,2 )
        binary_pred = binary_pred[: ,0 , : , : ]

        binary_pred = torch.tensor(binary_pred ) / 255

        mask_clone =  torch.where(d1_mask != 0, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32) ).to(device)
        binary_pred  =  torch.where(binary_pred  != 0,  torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))

        foreground_mask = (d1_mask > 0).float().to(device)


      else : # No refinement, other way of calling model :
        return print('Without refine model in such form is just simpleclassification, skipping for now... Results left for vizualization purpouses')
        
        # classification , coefs_p2 , loss , coarse_contours =  model()      
        # return classification , coefs_p2 , loss , coarse_contours # maybe add plot


      classification_map = classification_map.squeeze(1)

            #####       Custom loss calculation:       #######

      #FOREGROUND MASKS FOR CLASSIFICATION:


      #PENALTY FOR TOO MUCH DENDRITIC SPINES :
      binary_map = (classification_map >= 0.5).float()  # same thresh as in FULL network
      proportion = binary_map.mean()

      print(f'\nValue of proportion: {proportion} , shape of proportion: {proportion.shape}, penalty thresh is: {PENALTY_THRESH}\n')
      #Apply penalty if proportion exceeds threshold:
      penalty = torch.tensor([0.], dtype = torch.float , device = device)
      if proportion > PENALTY_THRESH :
        # print('P E N A L T Y !')
        penalty = PENALTY_WEIGHT  * (proportion - PENALTY_THRESH)
        penalty = penalty.to(device)

      ####   ####   QUANTITY PENALTY   ####   ####
      # Penalty for differnece in number of prediceted and marked instances

      quantity_penalty = torch.tensor([0.], dtype = torch.float , device = device)

      for idx in range(d2_mask.shape[0]) :  #  for mask in batch_size of masks  (in batch_size dim)

        #find empty point per image:
        empty_points = torch.all(coarse_contours[idx]  == 0 , dim = -1)
        empty_points = torch.all(empty_points , dim = - 1)
        pred_instances = torch.sum(~empty_points , dim = -1)

        marked_instances = torch.tensor(  len(torch.unique(d2_mask[idx]) ) , dtype = torch.float , device = device ) # num instances

        pred_instances = torch.tensor(  pred_instances , dtype = torch.float , device = device )

        print(f'For batch idx: {idx}. Number of marked_instances is equal to: {marked_instances} and number of pred_instances is equal to: {pred_instances}')

        gap = torch.log( torch.tensor([1.], device = device) + torch.abs(pred_instances - marked_instances)  )

        quantity_penalty += gap


      quantity_penalty = QUNATITY_PENALTY_WEIGHT * quantity_penalty

      # without weights:
      # 450 difference in log is like 2.65 , 100 difference is ~ 2. , 10 difference is 1 ,0 difference is 0

      #######################################################################################################

      # Apply foreground mask during loss computation to select only valid places for loss calculation:
      # Compute losses


      if P1_SHAPE == (1024,1024) :
        return f'P1_shape (1024,1024) was used only during test, for now skip. Adjustable'
        # # refined_contours = refined_contours.round()
        # refined_contours = smooth_contours(refined_contours = refined_contours , scale_factor= 2.0 )  

        # #   #White areas uint8:
        # classification_map = draw_refined_contours(contours = refined_contours , coarse = None , image_size = (1024,1024) , wdyw= 'area' , color = False )
        # classification_map = classification_map.to(torch.float) / 255.
        # classification_map = classification_map.permute(0,3, 1, 2 )

        # classification_map = classification_map[:,0 , : , : ].to(device)

        # foreground_mask = (mask > 0).float()

        # #Assign higher weight to foreground,  combine weights and mask
        # weights = torch.ones_like(classification_map , device = foreground_mask.device)
        # weights[mask > 0 ] = weights_object  # Foreground pixels get weight 5   # 10000000.0 -- Everything is class for i ==2
        # # d2_mask.unsqueeze(1)

        # combined_weights = weights * foreground_mask

        # classification_loss = torch.nn.functional.binary_cross_entropy(classification_map , d1_mask , weight= combined_weights )

        # # print(f'Shape of classification loss: {classification_loss.shape}')

        #   #FOREGROUND MASK FOR COEFS:

        # d1_coefs = get_fourier(d1_mask , max_num_coefs = MAX_NUM_COEFS , max_instances= MAX_NUM_INSTANCES , min_boundary_points= MIN_BOUNDARY_POINTS )

        # regression_loss = reg_loss(gt_coefs = d1_coefs , pred_coefs= coarse_predictions , device= device)   # d1_coef !!!!!!!!!  d2_coefs ?


      # FOR low reoslution computaitons
      elif P1_SHAPE == (512, 512) :

        foreground_mask = (d2_mask > 0).float()

        #Assign higher weight to foreground,  combine weights and mask   
        weights = torch.ones_like(classification_map , device = foreground_mask.device )
        weights[d2_mask > 0 ] = weights_object

        combined_weights = weights * foreground_mask

        classification_loss = torch.nn.functional.binary_cross_entropy(classification_map , d2_mask , weight= combined_weights )



        #FOREGROUND MASK FOR COEFS, take d1_coefs from gt_masks in higher spatial resolution:
        d1_coefs = get_fourier(d1_mask , max_num_coefs = MAX_NUM_COEFS , max_instances= MAX_NUM_INSTANCES , min_boundary_points= MIN_BOUNDARY_POINTS )

        regression_loss = reg_loss(gt_coefs = d1_coefs , pred_coefs= coarse_predictions , device= device)


      #USED ONLY GOR TESTS: 

      # print(f'Current classisifcation loss: {alpha * classification_loss.item()}')
      # print(f'Current reg coarse loss: {regression_loss.item()}')
      # print(f'Current refinment loss: {loss.item()}')    # int object has no .item() for no refine
      # print(f'Current class quantity penalty loss: {quantity_penalty.item()}')
      # print(f'Current penalty loss: {penalty.item()}\n\n')    # int object has no .item() for no refine

      # print(f'\nCurrent train accuracy loss: {train_acc}\n\n')  # deleted .item()


      # Backward pass and optimization
      loss = loss + alpha *classification_loss + regression_loss + penalty + quantity_penalty
      print_loss = alpha * classification_loss + regression_loss + penalty + quantity_penalty    # loss for comparisons training and testing part

      train_loss += loss.item()
      train_loss_print += print_loss.item()


      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    #Adjust metrics to get average loss per batch:
    train_loss /= len(train_dataloader)
    train_loss_print /= len(train_dataloader)

    #Additional metrics:
    # train_acc /= len(train_dataloader)


      ####  ####  ####  ####  ####  ####  ####  ####  ####  ####
    ####  ####  ####  ####  ####  ####  ####  ####  ####  ####

              #####     TEST PART:     #####

      ####  ####  ####  ####  ####  ####  ####  ####  ####  ####
    ####  ####  ####  ####  ####  ####  ####  ####  ####  ####

    #Put model in evaluation mode:
    model.eval()

      #Turn on inference mode:
    with torch.inference_mode() :

      # Loop through DataLoader:
      for img, mask  , d2_mask , d2_coefs  , d1_mask ,  in tqdm(test_dataloader) :


        #DEVICES
        img , mask , d2_mask , d2_coefs , d1_mask  = img.to(device) , mask.to(device) , d2_mask.to(device) , d2_coefs.to(device) , d1_mask.to(device)

        model.eval()

        # final_mask , rgb_mask  = model(img , device = device , )   # NONE   # gt_d1_mask =  None


        refined_contours ,  instance_mask , draw   = model(img ,gt_d1_mask =  None, test_mode = True, sigma = sigma , num_iter = num_iter,  device = device ,  p1_shape = P1_SHAPE )   # NONE   # gt_d1_mask =  None
        # final_mask , rgb_mask  = model(img , no_refine = True , test_mode = True , device = device , )   # NONE   # gt_d1_mask =  None


        d1_mask= torch.nn.functional.interpolate( d1_mask.unsqueeze(1).to(torch.float), scale_factor= 2, mode="bilinear", align_corners=True)
        d1_mask = d1_mask.squeeze(1)

        mask_clone =  torch.where(d1_mask != 0, 1, 0 ).to(device)



        binary_pred = draw_refined_contours(refined_contours , image_size = (1024, 1024) , wdyw ='area' , color = False)
        binary_pred  =  torch.where( binary_pred  != 0,  1, 0 ).to(device)
        binary_pred = binary_pred[: , : , : , 0]

        print(f'Shape of binary_pred: {binary_pred.shape}')


        foreground_mask = (d1_mask > 0).float().to(device)

        foreground_mask = foreground_mask.squeeze(1)
        print(f'Shape of foregorundmask: {foreground_mask.shape}')


        correct = (binary_pred == mask_clone) * foreground_mask  # Only count foreground matches
        print(f'MIDDLE correct shape: {correct.shape}')

        total = foreground_mask.sum().item()  # Count only foreground pixels

        print(f'Unique binary pred and mask clone: {torch.unique(binary_pred)} {torch.unique(mask_clone)}')
        print(f'Sum of == values (before masking): {(binary_pred == mask_clone).sum().item()}')
        print(f'Sum of == values after foreground mask: {correct.sum().item()}')
        print(f'Total foreground pixels: {total}')
        print(f'Accuracy over foreground pixels: {correct.sum().item() / (total + 1e-6)}')  # Avoid division by zero

        print('CORRECT SHAPE:' , correct.shape)

        # Move tensors to CPU and detach them for plotting
        foreground_mask_cpu = foreground_mask[0].cpu().squeeze().numpy()
        mask_clone_cpu = mask_clone[0].cpu().squeeze().numpy()

        print(f'shapes all: foregroundd maks one: {foreground_mask_cpu.shape} , mask clone one: {mask_clone_cpu.shape}')
        print(f'correct one: {correct[0].shape} binary pred one: {binary_pred[0].shape}')
        correct_cpu = correct[0].cpu().squeeze().numpy()
        binary_pred_cpu = binary_pred[0].cpu().squeeze().numpy()


        test_acc = correct.sum().float() / total  #if total > 0 else torch.tensor(0.0)


        print(f'\nTest  acc is: {test_acc} \n\n')

        test_acc += test_acc


        print('----------TESTING --------')


        # Create foreground mask from ground truth classification d2_mask :
        foreground_mask = (mask > 0).float()  # Binary mask: 1 for object pixels, 0 for background

        # Apply foreground mask during loss computation to select only valid places for loss calculation:

        foreground_mask = foreground_mask.to(device)

                #   #White areas uint8:
        classification_map = draw_refined_contours(contours = refined_contours.to(device) , coarse = None , image_size = (1024,1024) , wdyw= 'area' , color = False )
        classification_map = classification_map.to(torch.float) / 255.

        #PENALTY FOR TOO MUCH DENDRITIC SPINES :    
        proportion = classification_map.mean()

        print(f'\nValue of proportion: {proportion} , shape of proportion: {proportion.shape}, penalty thresh is: {PENALTY_THRESH}\n')
        #Apply penalty if proportion exceeds threshold:
        penalty = torch.tensor([0.], dtype = torch.float , device = device)
        if proportion > PENALTY_THRESH :
          # print('P E N A L T Y !')
          penalty = PENALTY_WEIGHT  * (proportion - PENALTY_THRESH)
          penalty = penalty.to(device)


                ####   ####   QUANTITY PENALTY   ####   ####
        # Penalty for differnece in number of prediceted and marked instances

        quantity_penalty = torch.tensor([0.], dtype = torch.float , device = device)

        for idx in range(mask.shape[0]) :  #  for mask in batch_size of masks  (in batch_size dim)


          # print(refined_contours[idx].shape , 'here is shape of refined.')

          #find empty point per image:
          empty_points = torch.all(refined_contours[idx]  == 0 , dim = -1)
          empty_points = torch.all(empty_points , dim = - 1)
          pred_instances = torch.sum(~empty_points , dim = -1)

          marked_instances = torch.tensor(  len(torch.unique(mask[idx]) ) , dtype = torch.float , device = device ) # num instances

          pred_instances = torch.tensor(  pred_instances , dtype = torch.float , device = device )

          print(f'For batch idx: {idx}. Number of marked_instances is equal to: {marked_instances} and number of pred_instances is equal to: {pred_instances}')

          gap = torch.log( torch.tensor([1.], device = device) + torch.abs(pred_instances - marked_instances)  )

          quantity_penalty += gap


        quantity_penalty = QUNATITY_PENALTY_WEIGHT * quantity_penalty

        # without weights:
        # 450 difference in log is like 2.65 , 100 difference is ~ 2. , 10 difference is 1 ,0 difference is 0

        #######################################################################################################

        classification_map = classification_map.permute(0,3,1,2 )   # BS , CC , H , W
        # classification_map = classification_map[0 , : , : ].unsqueeze(0).to(device)
        classification_map = classification_map[:,0 , : , : ].to(device)

        #Assign higher weight to foreground,  combine weights and mask
        weights = torch.ones_like(classification_map , device = classification_map.device )
        weights[mask > 0] = weights_object  # Foreground pixels get weight 5   # 10000000.0 -- Everything is class for i ==2

        combined_weights = weights * foreground_mask

        # Compute losses:
        classification_loss = torch.nn.functional.binary_cross_entropy(classification_map , mask ,weight= combined_weights )


        mask_coef = get_fourier(mask= mask , max_num_coefs = MAX_NUM_COEFS , max_instances= MAX_NUM_INSTANCES ,min_boundary_points= MIN_BOUNDARY_POINTS)
        final_coef = get_fourier(mask = classification_map , max_num_coefs = MAX_NUM_COEFS , max_instances= MAX_NUM_INSTANCES ,min_boundary_points= MIN_BOUNDARY_POINTS)

            #FOREGROUND MASK FOR COEFS:

        regression_loss = reg_loss(gt_coefs = mask_coef , pred_coefs=  final_coef , device = device )


        #USED ONLY FOR TESTS: 
        print(f'Current classisifcation loss: {alpha * classification_loss.item()}')
        print(f'Current reg coarse loss: {regression_loss.item()}')
        print(f'Current class quantity penalty loss: {quantity_penalty.item()}')
        print(f'Current penalty loss: {penalty.item()}\n\n')    

        # Backward pass and optimization
        loss = 0 + alpha * classification_loss + regression_loss +  quantity_penalty + penalty # loss
        test_loss += loss.item()

        print(f"Current, batch loss: {loss.item() }")   # later for skip


      test_loss /= len(test_dataloader)

      test_acc /= len(test_dataloader)


    ######    PRINTS AND CHECKPOINTS:   ######


    print(f'\n\nEpoch: {epoch} | Complete train loss with refinement: {train_loss:.4f} | Train loss: {train_loss_print:.4f}  | Test loss: {test_loss:.4f} |\n\n')

    # if (epoch) % 2 == 0   :  #  to start from next one epoch
      # save_checkpoint(model, optimizer, epoch , filename=f'{model_name}_checkpoint.pth' )

    # Update results dictionary:
    results['train_loss'].append(train_loss)
    results['train_loss_no_ref'].append(train_loss_print)
    results['test_loss'].append(test_loss)

    results['test_acc'].append(test_acc.item())

    ### Experiment tracking:
    if writer :

      writer.add_scalars(main_tag = 'Loss' ,
                        tag_scalar_dict = {'train_loss': train_loss ,
                                           'train_loss_no_ref' : train_loss_print ,
                                            'test_loss' : test_loss} ,
                        global_step = epoch )

      #Close the writer:
      writer.close()

    else :
      pass

  # Save checkpoint after every epoch to possible continuation in model training:
  save_checkpoint(model, optimizer, epoch , filename=f'{model_name}_checkpoint.pth' )

  # Save the model state dict:
  save_model(model=model,
            target_path= 'models',
            model_name= f'{model_name}_{epochs}_epochs.pth')

  # Return the filled results at the end of the epoch:
  # Those reults might be used to plot loss curve graphs
  return results


