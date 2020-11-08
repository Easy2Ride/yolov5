import os
import cv2
import pickle
import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot
# from torchsummary import summary


from utils.utils import *
from utils.datasets import *
from models.yolo import Model

from models.models import Darknet
from mono_ins import Monodepth2

from fastdepth_api import MobileNetSkipAdd

device = torch.device(device='cpu' if not torch.cuda.is_available() else 'cuda:0') 

class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear')

class Preprocess(nn.Module):
    def __init__(self,
                 calib_size = (416, 256),
                 resize_mode = 'INTER_AREA',
                 interpolation = cv2.INTER_AREA,
                 calib_dir = "./calibration/"):
        super(Preprocess, self).__init__()
        
        # Here we load default calibration matrix into map
        mtx, dist, dimension = pickle.load(open(os.path.join(calib_dir, resize_mode + "_" + str(calib_size[0]) + "_" + str(calib_size[1]) + ".pkl"), 'rb'))
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, dimension, cv2.CV_32F)
        
        map1 = 2 * map1 / calib_size[0] - 1
        map2 = 2 * map2 / calib_size[1] - 1

        calib_map = np.stack([map1, map2], axis = -1)
        self.map = torch.from_numpy(calib_map).unsqueeze(0).to(device)
                
    def forward(self, x, calib_map = None):
        if calib_map == None:
            calib_map = self.map 
        # Preprocess    
        x = x.flip(2)
        x = F.grid_sample(x, calib_map, mode = 'bicubic')
        
        return x
                            

class DepthDetector(nn.Module):
    def __init__(self,
                 config_path='models/yolov5s.yaml',
                 img_hw = (480, 640),
                 detect_hw = (256, 416),
                 depth_hw = (192, 640),
                 pretrained = True, 
                 mobile_encoder_path = './monodepth_weights/mobilenet_best.pth',
                 num_ch_enc = np.array([64, 64, 128, 256, 512]), 
                 num_ch_dec = np.array([16, 32, 64, 128, 256]), 
                 num_input_images = 1, 
                 num_output_channels = 1, 
                 scales = range(4), 
                 use_skips = True,
                 arch = 'yolov3',
                 cls_act = 'sigmoid',
                 onnx_flag = True,
                 use_exp = False,
                 ):
        super(DepthDetector, self).__init__()
        if isinstance(img_hw, int):
            img_hw = (img_hw, img_hw)
                            
        self.img_hw = img_hw
        self.detect_hw = detect_hw
        self.depth_hw = depth_hw
        self.depth = Monodepth2(pretrained, num_ch_enc, num_ch_dec, num_input_images, num_output_channels, scales, use_skips)

        #self.depth = MobileNetSkipAdd((self.depth_hw[1], self.depth_hw[0]), pretrained, pretrained_path = mobile_encoder_path, scales = scales)
        #self.detector = Darknet(config_path, self.detect_hw, arch = arch, cls_act = cls_act, onnx_flag = onnx_flag, use_exp = use_exp)
        
        self.detector = Model(config_path)
        self.depth.to(device).eval()
        self.detector.to(device).eval()
        
        
    def forward(self, x):
        # Preprocess
        # x /= 255.0                                                          # Normalization
        # x = x[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()              # BGR to RGB
        
        if self.detect_hw != self.img_hw:
        	x1 = Resize(self.detect_hw)(x)
        else:
            x1 = x
        
        if self.depth_hw != self.detect_hw:
            x2 = Resize(self.depth_hw)(x)
        else:
            x2 = x1
        
        # Detect objects
        x1 = self.detector(x1)
        
        # Estimate depth
        x2, x3, x4 = self.depth(x2)
        x2 = x2.view(-1, 1, self.depth_hw[0], self.depth_hw[1])
        # x2 = Resize(self.detect_hw)(x2)
        
        #return x1[0], x1[1], x2
        return x1, x2, x3, x4

#[torch.Size([1, 16, 16, 26]), torch.Size([1, 4, 16, 26]), torch.Size([1, 1, 192, 320])]
        

if __name__ == '__main__': 
    save_onnx = True
    use_exp = False
    cls_act = 'sigmoid'
    arch = 'resnet'

    # Change this if you want input (height, width) to be same as detection (height, width)
    #img_hw = (480, 640)
    # img_hw = (256, 416)
    # detect_hw = (256, 416)
    # depth_hw = (128, 224)
    img_hw = (256, 416)
    detect_hw = (256, 416)
    depth_hw = (160, 288)
    shape = (3,) + img_hw 
    
    # Uncomment to use yolov3
    #config_path = 'cfg/yolov3_custom.cfg'
    #weights_path = './weights/best.pt'
    
    # Uncomment to use resnet10 (also comment/uncomment inside the model to use darknet instead)
    #config_path = 'cfg/resnet10.cfg'
    #weights_path = './weights/resnet10.pt'
    
    # Uncomment to use yolov5s
    config_path = 'models/yolov5s.yaml'
    weights_path = 'weights/yolov5s.pt'
    
    # Common for all (but comment/uncomment required inside the model for fastdepth/monodepth or yolov5/resnet10 selection)
    num_ch_enc = np.array([64, 64, 128, 256, 512])
    num_ch_dec = np.array([16, 32, 64, 128, 256])
    model_path = './monodepth_weights/'
    encoder_path = os.path.abspath(os.path.join(model_path, "encoder.pth"))
    depth_decoder_path = os.path.abspath(os.path.join(model_path, "depth.pth"))
    mask_decoder_path = os.path.join(model_path, "mask.pth")
    mobile_encoder_path = os.path.abspath(os.path.join(model_path, "../mobilenet_best.pth"))
    model = DepthDetector(config_path, 
                         img_hw, 
                         detect_hw,
                         depth_hw,
                         pretrained = True, 
                         mobile_encoder_path = mobile_encoder_path,
                         num_ch_enc = num_ch_enc, 
                         num_ch_dec = num_ch_dec, 
                         arch = arch, 
                         cls_act = cls_act, 
                         onnx_flag = save_onnx, 
                         use_exp = use_exp)

    # Uncomment for yolov5s
    with open('data/coco128.yaml') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    model.detector.names = data_dict['names']
    ckpt = torch.load(weights_path, map_location=device)
    ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if model.detector.state_dict()[k].shape == v.shape}
    model.detector.load_state_dict(ckpt['model'], strict=False)
    model.detector.fuse() 
    
    # Uncomment for resnet10
    #model.detector.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    
    model.to(device)
    model.eval()
    #summary(model, shape)    
    

    # # Uncomment the following for monodepth as depth estimation model   
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract the height and width of image that this model was trained with
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in model.depth.encoder.state_dict()}
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    mask_loaded_dict = torch.load(mask_decoder_path, map_location=device)
    model.depth.encoder.load_state_dict(filtered_dict_enc)
    model.depth.depth_decoder.load_state_dict(loaded_dict)
    model.depth.mask_decoder.load_state_dict(mask_loaded_dict)


    
    # Uncomment the following for fastdepth as depth estimation model
    #loaded_dict_enc = torch.load(os.path.join(model_path,"fastdepth.pth"), map_location=device)
    #model.depth.load_state_dict(loaded_dict_enc)
    
    #ckpt = {'model': model.state_dict()}
    #torch.save(ckpt, 'mono_yolov5.pt')
    
    x = torch.zeros((1,) + shape).to(device)
    #x1, x2 = Preprocess()(x)
    
    # x1 = torch.zeros((1,) + (3,) + detect_hw).to(device)
    # x2 = torch.zeros((1,) + (3,) + depth_hw).to(device)
    for _ in range(10):
        t1 = time.time()
        y = model(x)
        print(time.time() - t1)
    #print([x.shape for x in y])
    
    if save_onnx:
        # save_path = "../combined_model_benchmark/Onnx/mono_yolo_640.onnx"
        save_path = "/home/dnaish/Documents/Algo Design/Build_Oct_2020/trt_pipeline_all/models/mono_ins_yolov5_oct_demo.onnx"
        save_path = "./mono_ins_yolov5_oct_demo.onnx"
        print("Saving onnx file to {} ...".format(save_path))
        print(x.shape)
        # x = [x1, x2] #torch.zeros((1,) + shape).to(device)
        input_names = ["image_input"]
        output_names = [ "bbox_output", "depth_output", "seg_mask_output", "ins_mask_output"]
        # Uncomment for resnet10
        #output_names = [ "conv2d_bbox", "conv2d_cov/Sigmoid", "depth_output"]
        torch_out = torch.onnx._export(model,                       # model being run
                                       x,                           # model input (or a tuple for multiple inputs)
                                       save_path,                   # where to save the model (can be a file or file-like object)
                                       export_params = True,        # store the trained parameter weights inside the model file
                                       do_constant_folding = True,
                                       input_names = input_names, 
                                       output_names = output_names,
                                       keep_initializers_as_inputs = True)      
                                       
    print('Checking onnx model')
    import onnx
    model = onnx.load(save_path)
    onnx.checker.check_model(model)
    print(save_path + '    onnx model generated ')
