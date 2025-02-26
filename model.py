
import torch
# import torchvision
import torch.nn as nn

from utils import create_instance_map , get_fourier , make_pixel , draw_refined_contours , refine 


class BackboneNN(nn.Module) :
  def __init__(self , backbone , ) :
    super(BackboneNN , self ).__init__()
    self.backbone = backbone  # Pretrained NN (try different ResNet's and others feature extractors )


  def forward(self , x ) -> torch.tensor :
    # Pretrained network used as bacbone to read feature maps: high and low

    # For Resnet50:
    layers = list(self.backbone.children())
    p2_features = torch.nn.Sequential(*layers[:6])  # Low resolution Resnet50
    p2_features = p2_features(x)

    p1_features = torch.nn.functional.interpolate( p2_features , size=( 512, 512 ),  mode='bilinear', align_corners=False ) # (sth , sth , 256, 256) --> (sth, sth , 512, 512)

    #SO whats left by default is for p2_f and p1_features: torch.Size([BS, 512, 256 , 256]) torch.Size([BS, 512, 512 , 512])
    return p1_features , p2_features


class ContourProposalNetwork(nn.Module):
  def __init__(self, backbone, num_coefs : int = 20 , num_instances : int = 600 ,
                p2_channels : int = 512  , p1_features : int = 512   ) :

      '''  Network for contour proposal and refinement.
      Args:
      backbone: A backbone network that outputs two feature maps.
      num_coefs: Number of Fourier coefficients for higher-resolution features == lower-resolution features
      num_instances: maximal number of instances per image
      p2_channels: input channels of p2_resolution. Must be correctly set with rest of arguments
      p1_features: Number of features in higher spatial resolution. Must be correctly set with rest of arguments'''

      super().__init__()
      self.backbone = backbone  # Backbone to extract features

      self.num_coefs = num_coefs
      self.num_instances = num_instances
      self.p2_channels = p2_channels

      self.p1_features = p1_features


      # Head for binary classification (foreground/background)
      self.classification_head = nn.Sequential(
          nn.Conv2d( self.p2_channels , 1024, kernel_size=3, padding=1),  # Input: lower-res features
          nn.ReLU(),
          nn.Conv2d( 1024 , 512, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d( 512, 256, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(256, 1, kernel_size=1),  # Output: (B, 1, H2, W2)
          nn.Sigmoid()  # Binary classification
      )


      self.refinement = nn.Sequential(
          nn.Conv2d(self.p1_features ,  1024, kernel_size=3, padding=1) ,
          nn.ReLU() ,
          nn.Conv2d(1024 , 512 , kernel_size = 3 , padding = 1 ) ,
          nn.ReLU() ,
          nn.Conv2d(512 , 256 , kernel_size = 3 , padding = 1) ,
          nn.ReLU() ,
          nn.Conv2d(256 ,  2 , kernel_size=1),  # # Outputs (dx, dy) field [x,y] --> num_coef back
      )

    # IF you give p1_features, then we perform refinement.. Otherwise skip


  def forward(self, x ,device ,  p1_shape : tuple = (512, 512) ,  gt_d1_mask = None , sigma = 1. , num_iter : int = 2 , no_refine = False , test_mode = False ):
      """ Args: x == Input image tensor, shape (B, C, H, W)
      Returns:
          classification: Binary masks, shape (B, 1, H2, W2)
          contour: Fourier coefficients, shape (B, num_coefs_2, H2, W2)
          refinement: Refinement coefficients, shape (B, num_coefs_1, H1, W1)
          gt_d1_mask: pass to perfrome refinement 
          sigme: multiplier of refine step possible movement 
          num_iter: number of refinement iterations 
          no_refine: By default set to False, which means performe refine
          test_mode: By default False, to evalution mode set to true. 
      """

      # Backbone extracts features
      features_1, features_2  = self.backbone(x)  # Features at lower resolution
      features_1 , features_2  = features_1.to(device) , features_2.to(device)

      if  gt_d1_mask is not None  :
        gt_d1_mask = gt_d1_mask.to(device)

      # Predictions from the lower-res binary mask
      classification = self.classification_head(features_2)

      binary_map = (classification >= 0.5).float()
      # :use for mask reconstruction, might be diffrent threshold like 0.5 or above. IMPORTANT


      # Apply bilinear interpolation
      binary_map= torch.nn.functional.interpolate( binary_map.to(torch.float), scale_factor= 2, mode="bilinear", align_corners=True)
      binary_map = binary_map.round()


      #Turn it into instance map of max_num_instances biggest examples:
      selected_map = create_instance_map(binary_map.squeeze(1) , max_num_instances= self.num_instances )

      coefs_p2 = get_fourier(mask = selected_map, max_num_coefs = self.num_coefs , max_instances= self.num_instances , min_boundary_points = 2).to(device)
      # SHape [BS ,  num_instances , num_coefs ]


      #Move coefficient (B, max_num_instances, max_coefs)---> pixel space for refinement later:
      draw , coarse_contours = make_pixel(coefficients = coefs_p2 , img_shape = (binary_map.shape[2] , binary_map.shape[3]) , wdyw = 'area' )


      if gt_d1_mask is not None and no_refine == False :

        # Refinmnet loss:
        gt_d1_mask = gt_d1_mask.to(device)
        gt_contours = get_fourier(gt_d1_mask).to(device) # get coefs from ideal mask
        draw , gt_contours = make_pixel(coefficients= gt_contours , img_shape=(p1_shape) , wdyw = 'area')

        loss = 0

        # Predict refinement field (dx, dy) from image
        refinement_field = self.refinement(features_1 )  # Shape: (B, 2, H, W)

        # # Clone input to refine iteratively
        refined_contours = coarse_contours.clone()
        refined_contours = refined_contours.to(device)


        for iter_idx in range(num_iter):
          # print(f"\n--- Iteration {iter_idx + 1} ---")
          loss , refined_contours = refine(refined_contours = refined_contours , mask_contours = coarse_contours , gt_contours = gt_contours , refinement_field = refinement_field , img_shape = p1_shape , sigma = sigma )

          # Accumulate loss from each step, for improvement:
          loss += loss


        return classification.to(device), coefs_p2.to(device)  , loss.to(device) , refined_contours.round()  # --> [Bs , 1, H , W] QQQ2 final_mask


      ####  TRAINING WITHOUT REFINE:
      elif no_refine == True and test_mode == False :
        #Just returning refine loss as 0 + no looping cycle.
        loss = torch.tensor([0.], dtype = torch.float , device = device)

        return classification.to(device) , coefs_p2.to(device)  , loss , coarse_contours



      ###########  E V A L U A T I O N  ##############

      # FOR TESTING SET test_mode == TRUE
      if test_mode == True :
        #EVALUATION WITH REFINE:
        if gt_d1_mask is None and no_refine == False :

          print('EVALUATION TIME!!')

          refinement_field = self.refinement(features_1)# # Clone input to refine iteratively
          refined_contours = coarse_contours.clone()
          refined_contours = refined_contours.to(device)

          for iter_idx in range(num_iter):
            # Here just apply refinment, no loss accumulation
            refined_contours = refine(refined_contours = refined_contours , mask_contours = coarse_contours ,
                                          refinement_field = refinement_field ,
                                          img_shape = p1_shape  , sigma = sigma )    # no gt


          if p1_shape == (512,512) :

            draw = draw_refined_contours(contours = refined_contours , image_size = (512 ,512) , color = False , wdyw = 'area' )

            draw= draw.permute(0 , 3 , 1 , 2)
            draw = draw[: , 0 ,  : , : ]
            draw = draw.unsqueeze(1)

            draw = torch.nn.functional.interpolate( draw.to(torch.float), scale_factor= 2, mode="bilinear", align_corners=True)
            draw = draw.round()

            draw = draw.squeeze(1)

            coefs_p1 = get_fourier(mask = draw, max_num_coefs = self.num_coefs , max_instances= self.num_instances , min_boundary_points = 2).to(device)

            instance_mask = create_instance_map(draw , max_num_instances = self.num_instances )


            draw ,refined_contours = make_pixel(coefficients = coefs_p1 , img_shape = (1024, 1024) , wdyw = 'area'  ) # here draw is not important


            draw = draw_refined_contours(contours = refined_contours , coarse = None , image_size = (1024,1024) , wdyw= 'contours' , color = True )

            return refined_contours ,  instance_mask , draw

          # elif p1_shape == (1024, 1024) :
          #   # No refine rescaling is necessary
          #   refined_contours = refined_contours.round()

          #   #White areas uint8:
          #   draw = draw_refined_contours(contours = refined_contours , coarse = None , image_size = (1024,1024) , wdyw= 'area' , color = False )

          #   # get correct shape and gain biggest instances by default:
          #   draw= draw.permute(0 , 3 , 1 , 2)
          #   draw = draw[: , 0 ,  : , : ]
          #   draw = draw.squeeze(1)

          #   instance_mask = create_instance_map(draw , max_num_instances = self.num_instances )

          #   refined_contours = get_fourier(instance_mask.squeeze(1) , max_num_coefs = self.num_coefs , max_instances= self.num_instances , min_boundary_points = 2)
          #   _ , refined_contours = make_pixel(refined_contours , img_shape = (1024, 1024) , wdyw = 'contours')

          #   draw = draw_refined_contours(contours = refined_contours , coarse = None , image_size = (1024,1024) , wdyw= 'contours' , color = True )

          #   return refined_contours ,  instance_mask , draw


        # EVALUATION WITHOUT REFINE :   SKIPPED Couse of no rellevant result for simple 
        else :
          if p1_shape == (512,512) :
            return 'MODIFIED PART. SKIPPED '
            # return  coarse_contours ,  instance_mask , draw


          elif p1_shape == (1024, 1024) :
            return 'MODIFIED PART. SKIPPED '
         

