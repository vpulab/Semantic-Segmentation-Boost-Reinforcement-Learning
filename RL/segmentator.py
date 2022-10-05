
import torch

from torchvision.models.segmentation.deeplabv3 import DeepLabHead,DeepLabV3
from torchvision import models,transforms

import numpy as np
from PIL import Image



class Segmentator():

    def __init__(self,model_name='ResNet50.pth'):
        segmodel = models.segmentation.deeplabv3_resnet50(
                pretrained=True, progress=True)
        # Added a Sigmoid activation after the last convolution layer
        segmodel.classifier = DeepLabHead(2048, 6)###
        #print(segmodel) # if you want, you can visually inspect the defined architecture
        transform_input = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean, std),
        ])
        segmodel = segmodel.to('cuda')
        segmodel.load_state_dict(torch.load('Segmentation_model/'+model_name))
        segmodel.eval()
        self.model = segmodel
        self.transform_input = transform_input


    def decode_segmap(image, nc=21):
    ## Color palette for visualization 
        label_colors = np.array([(0, 0, 0),  # 0=background
                    (0, 0,255), (127, 127, 0), (127, 127, 0), (255, 0, 0), (255, 255, 0),
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb

   
    def segment(self, frame,transform=None, dev='cuda'):
        img = frame
        transform = self.transform_input
        if torch.cuda.is_available():
                input_image = transform(img).unsqueeze(0).cuda()
        
        out = self.model(input_image)['out'][0]
        
        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        segm_rgb = Segmentator.decode_segmap(segm)
        return segm_rgb
    
    def segment_labels(self, frame,transform=None, dev='cuda'):
        img = frame
        transform = self.transform_input
        if torch.cuda.is_available():
                input_image = transform(img).unsqueeze(0).cuda()
        
        out = self.model(input_image)['out'][0]
        
        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        return segm