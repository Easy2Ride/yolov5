import argparse
import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchsummary import summary
from time import time

save_onnx = True
#########################################################

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, pretrained, num_ch_enc = np.array([64, 64, 128, 256, 512]), num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.encoder = models.resnet18(pretrained)
       

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec = np.array([16, 32, 64, 128, 256]), scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        input_features = [feat1, feat2, feat3, feat4, feat5]
        self.outputs = [] # {}
        # decoder
        x = input_features[-1]
        
        for i in range(4, -1, -1):
        
            x = self.convs[("upconv", i, 0)](x)
            
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            #if i in self.scales:
                # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            #    self.outputs.append(self.sigmoid(self.convs[("dispconv", i)](x)))
            
            if i == 0:
                # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                return self.sigmoid(self.convs[("dispconv", i)](x))
        # return self.outputs

class MaskDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=20, use_skips=True, n_objects = 64):
        super(MaskDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.n_objects = n_objects

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("maskconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            
            
        self.convs[("insconv", 0)] = Conv3x3(self.num_ch_dec[0], self.n_objects)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, feat1, feat2, feat3, feat4, feat5):
        input_features = [feat1, feat2, feat3, feat4, feat5]
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("seg_mask", i)] = self.convs[("maskconv", i)](x)
                if i == 0:
                    self.outputs[("ins_mask", i)] = self.convs[("insconv", i)](x)
                    return self.outputs[("seg_mask", 0)], self.outputs[("ins_mask", 0)]
                
        return self.outputs
                
class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='nearest')
        
class Monodepth2(nn.Module):
    def __init__(self, 
                 pretrained, 
                 num_ch_enc = np.array([64, 64, 128, 256, 512]), 
                 num_ch_dec = np.array([16, 32, 64, 128, 256]), 
                 num_input_images=1, 
                 num_output_channels=1, 
                 scales=range(4), 
                 use_skips=True,
                 input_size = (160, 288)):
        super(Monodepth2, self).__init__()
        self.encoder = ResnetEncoder(pretrained)
        self.depth_decoder = DepthDecoder(num_ch_enc, num_ch_dec = num_ch_dec, scales = scales)
        self.mask_decoder = MaskDecoder(num_ch_enc, num_output_channels = 20, n_objects = 64)
        self.resize = Resize(input_size)
        
    def forward(self, x):
        x = self.resize(x)
        x = self.encoder(x)
        x1 = self.depth_decoder(*x)
        x2, x3 = self.mask_decoder(*x)
        output = [x1, x2, x3]
        return output
        

def disp_to_depth(disp, min_depth, max_depth):#min_depth = 0.0079, max_depth = -1
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

    return scaled_disp, depth

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=False):
        super(Conv3x3, self).__init__()
        # Not available in TensorRT
        '''if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)'''
        #  self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, padding=1, padding_mode="zeros")
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, padding=1, padding_mode='zeros')
    def forward(self, x):
        # x = self.pad(x)
        out = self.conv(x)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    # See: https://github.com/onnx/onnx-tensorrt/issues/192
    sh = list(x.shape)
    rsize = (int(sh[2] * 2), int(sh[3] * 2))
    return F.interpolate(x, size=rsize, mode='nearest')
    # size=(sh[2] * 2, sh[3] * 2),
    # return F.interpolate(x, scale_factor=2, mode="nearest")

def predict_depth(input_image):
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    with torch.no_grad():
        input_image = input_image.to(device)
        # features = encoder(input_image)
        # outputs = depth_decoder(*features)
        outputs = model(input_image)
        disp = outputs[0]  # [("disp", 0)]
        # camera parameter already added to the function
        scaled_disp, res_depth = disp_to_depth(disp, 0.1, 100)
        scaled_disp = torch.nn.functional.interpolate(scaled_disp, (192, 640), mode="bilinear", align_corners=False)
     
    return scaled_disp.squeeze().cpu().numpy(), res_depth.squeeze().cpu().numpy()#disp, scaleyoutuberes_depth
    

device = torch.device(device='cpu' if not torch.cuda.is_available() else 'cuda:0')
num_ch_enc = np.array([64, 64, 128, 256, 512])
num_ch_dec = np.array([16, 32, 64, 128, 256])
model = Monodepth2(pretrained = False, num_ch_enc = num_ch_enc, num_ch_dec = num_ch_dec)
model.to(device)
# summary(model, (3, 192, 640))
    
    
def load_pretrained(model_path = '../combined_model_benchmark/Pytorch'):
    #########################################################
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    mask_decoder_path = os.path.join(model_path, "mask.pth")
    # LOADING PRETRAINED MODEL

    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in model.encoder.state_dict()}
    depth_loaded_dict = torch.load(depth_decoder_path, map_location=device)
    mask_loaded_dict = torch.load(mask_decoder_path, map_location=device)
    model.encoder.load_state_dict(filtered_dict_enc)
    # model.depth_decoder.load_state_dict(depth_loaded_dict)
    model.mask_decoder.load_state_dict(mask_loaded_dict)
    print('loaded weights')
    if save_onnx:
        save_path = "../combined_model_benchmark/Onnx/monodepth2.onnx"
        save_path = "sem_ins.onnx"
        print("Saving onnx file to {} ...".format(save_path))
        input_names = ["image_input"]
        output_names = ["depth_output", "seg_mask_output", "ins_mask_output"] #, 
        x = torch.zeros((1, 3, 256, 416)).to(device)
        torch_out = torch.onnx._export(model,                       # model being run
                                       x,                           # model input (or a tuple for multiple inputs)
                                       save_path,                   # where to save the model (can be a file or file-like object)
                                       export_params = True,        # store the trained parameter weights inside the model file
                                       do_constant_folding = True,
                                       keep_initializers_as_inputs = True
                                      )    
    return model  
                                   

# summary(depth_decoder, [(64, 208, 208),  (64, 104, 104), (128, 52, 52), (256, 26, 26), (512, 13, 13)])

'''
encoder = ResnetEncoder(False)
depth_decoder = DepthDecoder(num_ch_enc=num_ch_enc, scales=range(4))
encoder.to(device)
depth_decoder.to(device)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()
print("Loaded monodepth model")
'''



if __name__ == '__main__':
    from PIL import Image
    load_pretrained('monodepth_weights/')
    img = np.ones((192, 640, 3), dtype = np.uint8)
    img[:, :] = [1, 2, 3]
    for _ in range(5):
        t1 = time()
        predict_depth(img)
        t2 = time()
        print(t2 - t1)
