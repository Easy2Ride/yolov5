from darknetutils.google_utils import *
from darknetutils.layers import *
from darknetutils.parse_config import *
from torchsummary import summary
    
ONNX_EXPORT = False


device = torch.device(device='cpu' if not torch.cuda.is_available() else 'cuda:0')

def create_modules(module_defs, img_hw, cfg, arch = 'yolov3', cls_act = 'sigmoid', onnx_flag = False, use_exp = False):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_hw = [img_hw] * 2 if isinstance(img_hw, int) else img_hw  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU())
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'Scale':
            class CaffeScale(nn.Module):
                def __init__(self, in_channels, scale_init = 1., bias_init = 0.):
                    super(CaffeScale, self).__init__()
                    self.alpha = nn.Parameter(torch.ones(in_channels, device = device).fill_(scale_init))
                    self.beta = nn.Parameter(torch.zeros(in_channels, device = device).fill_(bias_init))
                def forward(self, x):
                    x = x * self.alpha.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
                    return x
            modules = CaffeScale(output_filters[-1])
            
        elif mdef['type'] == 'ReLU':
            modules = nn.ReLU()
            
        elif mdef['type'] == 'Sigmoid':
            modules = nn.Sigmoid()

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
                
        elif mdef['type'] == 'avgpool':
            k = mdef['size'] if 'size' in mdef else 2  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else k
            avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('AvgPool2d', avgpool)
            else:
                modules = avgpool

        elif mdef['type'] == 'upsample':
            class Upsample2d(nn.Module):
                def __init__(self, stride):
                    super(Upsample2d, self).__init__()
                    self.stride = stride
                def forward(self, x):
                    sh = list(x.shape)
                    rsize = (int(sh[2] * self.stride), int(sh[3] * self.stride))
                    return F.interpolate(x, size=rsize, mode='nearest')
            modules = Upsample2d(int(mdef['stride']))
            '''if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])'''

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
            
        elif mdef['type'] == 'scale_channels':             
            layers = mdef['from']
            routs.extend([i + l if l < 0 else l for l in layers])
            class ScaleChannels(nn.Module):
                def __init__(self, layers):
                    super(ScaleChannels, self).__init__()
                    self.layers = layers
                def forward(self, x, outputs):
                    for idx in range(len(self.layers)):
                        x_mult = outputs[self.layers[idx]]
                        sh = x_mult.shape
                        rsize = (int(sh[2]), int(sh[3]))
                        x = F.interpolate(x, size=rsize, mode='nearest')
                        x *= x_mult
                    return x
            modules = ScaleChannels(layers)
            filters = output_filters[-1]
                        
        elif mdef['type'] == 'dropout':  # yolov3-spp-pan-scale
            pass
            

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass
       
        elif mdef['type'] == 'reorg':
            class Reorg(nn.Module):
                def __init__(self, stride = 2):
                    super(Reorg, self).__init__()
                    self.stride = stride
                def forward(self, x):
                    stride = self.stride
                    if onnx_flag:
                        bs, ch, new_h, new_w = 1, 64, img_hw[0] // 32, img_hw[1] // 32
                    else:
                        bs, ch, h, w = x.shape
                        new_h, new_w = h // stride, w // stride
                    x = x.view(-1, ch, new_h, stride, new_w, stride).permute(0, 1, 3, 5, 2, 4).contiguous()
                    x = x.view(-1, ch * stride * stride, new_h, new_w)
                    return x
            stride = int(mdef['stride'])
            modules = Reorg(stride)
            filters = output_filters[-1] * stride * stride 

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]  # P5, P4, P3 strides
            if 'panet' in cfg or 'yolov4' in cfg:  # stride order reversed
                stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_hw=img_hw,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index],
                                arch = arch,
                                cls_act = cls_act,
                                onnx_flag = onnx_flag,
                                use_exp = use_exp)

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_hw, yolo_index, layers, stride, arch = 'yolov3', cls_act = 'sigmoid', onnx_flag = False, use_exp = False):
        super(YOLOLayer, self).__init__()

        if isinstance(img_hw, int):
            self.img_hw = (img_hw, img_hw)
        else:
            self.img_hw = img_hw
            
        self.anchors = torch.Tensor(anchors)
        self.anchors_vec = self.anchors / stride

        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.cls_act = cls_act
        
        self.arch = arch
        self.onnx_flag = onnx_flag
        self.use_exp = use_exp
        self.layers = layers  # model output layer 
        self.nl = len(layers)


        # if False:  # grids must be computed in __init__
        self.stride = stride  # stride of this layer
        self.nx = int(self.img_hw[1] / self.stride)  # number x grid points
        self.ny = int(self.img_hw[0] / self.stride)  # number y grid points
        # self.create_grids(img_hw, (nx, ny))
            
    def create_grids(self, ng=(13, 13), device='cpu', type=torch.float32):
        nx, ny = ng  # x and y grid size
        # nx, ny = self.nx, self.ny
        # self.img_hw = max(img_hw)
        # self.stride = self.img_hw / max(ng)
        
        # self.anchors *= torch.tensor(ng) * self.stride / 608

        # build xy offsets
        #yv = torch.arange(ny).unsqueeze(1).repeat(1, nx)
        #xv = torch.arange(nx).unsqueeze(0).repeat(ny, 1)
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny * nx, 2))
        self.anchor_wh = self.anchors.view(1, self.na, 1 * 1, 2).to(device).type(type)

        # build wh gains
        # self.ng = torch.Tensor(ng).to(device)
        # self.nx = nx
        # self.ny = ny

    def forward(self, p, out = None):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            self.create_grids((nx, ny), p.device)

            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        
        if self.onnx_flag:
            bs = 1  # batch size
            ny, nx = self.ny, self.nx
        else:
            bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]

        self.create_grids((nx, ny), p.device, p.dtype)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)

        
        if self.training:
            p = p.view(-1, self.na, 5 + self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            return p

        else:  # inference
            p = p.view(-1, self.na, 5 + self.nc, ny * nx).permute(0, 1, 3, 2).contiguous()
            x = p  # inference output
            
            x = torch.sigmoid(x)
            self.grid_xy = self.grid_xy.view(1, 1, ny * nx, 2).repeat(bs, self.na, 1, 1)
            self.anchor_wh = self.anchor_wh.view(1, self.na, 1, 2).repeat(bs, 1, ny * nx, 1)
            filler1 = torch.zeros(bs, self.na, ny * nx, 2, device = device)
            filler2 = torch.zeros(bs, self.na, ny * nx, 1 + self.nc, device = device)
            filler3 = torch.zeros(bs, self.na, ny * nx, 1 , device = device)
            filler4 = torch.ones(bs, self.na, ny * nx, 2, device = device)
            filler5 = torch.ones(bs, self.na, ny * nx, 1 + self.nc, device = device)
            filler6 = torch.ones(bs, self.na, ny * nx, self.nc , device = device)
            
            self.grid_xy = torch.cat([self.grid_xy, filler1, filler2], dim = 3)
            self.anchor_wh = torch.cat([filler4, self.anchor_wh, filler5], dim = 3)
            
            multiplier1 = torch.cat([filler4 * self.stride, filler4, filler5], dim = 3) 
            multiplier2 = torch.cat([filler4, 2 * filler4, filler5], dim = 3)
            selector1 = torch.cat([filler1, filler4, filler2], dim = 3)
            
            selector2 = torch.cat([filler1, filler1, filler3, filler6], dim = 3) 
            
            if self.use_exp or self.arch ==  'yolov2':
                x_select1 = x * selector1
                x_select1 = x_select1 / (1 - x_select1)
            else:
                x_select1 = ((x * selector1) * multiplier2) ** 3
            
            x = (x + self.grid_xy) * multiplier1
            x = (x * (1 - selector1) + x_select1) * self.anchor_wh
            
            if self.arch == 'yolov2':
                selector3 = torch.cat([filler4, filler4, filler2], dim = 3)
                multiplier3 = torch.cat([filler4, filler4 * self.stride, filler2], dim = 3)
                x = x * (1 - selector3) + x * multiplier3


            if self.nc == 1:
                x = x * (1 - selector2) + selector2 * 1
            elif self.cls_act == 'softmax' or self.arch == 'yolov2':
                x_select = selector2 * x
                x = x * (1 - selector2) + selector2 * torch.softmax(torch.log(x_select / (1 - x_select)), dim = 3)
            
            if self.onnx_flag:
                return x.view(-1, ny * nx * self.na * (5 + self.nc))
            else:
                return x.view(-1, ny * nx * self.na, 5 + self.nc), p
                
class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_hw=(416, 416), verbose=False, arch = 'yolov3', cls_act = 'sigmoid', onnx_flag = False, use_exp = False):
        super(Darknet, self).__init__()

        if isinstance(img_hw, int):
            img_hw = (img_hw, img_hw)
        self.img_hw = img_hw
        self.cls_act = cls_act
        self.onnx_flag = onnx_flag
        self.use_exp = use_exp
        self.arch = arch
        
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_hw, cfg, arch = arch, cls_act = cls_act, onnx_flag = onnx_flag, use_exp = use_exp)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        model_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        #if self.onnx_flag:
        #    x = F.interpolate(x, size=self.img_hw, mode='nearest')
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            # print(i, module)
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'ScaleChannels']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                model_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
                if self.arch == 'resnet' and i in  [len(self.module_list) - 1, len(self.module_list) - 4]:
                    model_out.append(x)
                    

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        	
        if self.training:  # train
            return model_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*model_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        elif self.arch == 'resnet':
            '''grid_dim = self.img_hw[0] // 16, self.img_hw[1] // 16
            if self.onnx_flag:
                bs = 1
            else:
                bs = model_out[0].shape[0]
            objectness = torch.ones((bs, grid_dim[0] * grid_dim[1] * 4, 1), device = device)
            bboxes = model_out[0].view(-1, grid_dim[0] * grid_dim[1] * 4, 4)
            scores = model_out[1].view(-1, grid_dim[0] * grid_dim[1], 4).repeat(1, 4, 1)
            print(bboxes.shape, scores.shape, objectness.shape)
            x = torch.cat([bboxes, objectness, scores], dim = 2)
            if self.onnx_flag:
                return x.view(bs, grid_dim[0] * grid_dim[1] * 4 * (5 + 4))'''
            return model_out      
        elif self.onnx_flag:
            x = torch.cat(model_out, 1)
            return x
        else:  # inference or test
            x, p = zip(*model_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training # change to int32 for weights with old headers (yolov2-coco, resnet10)

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    print('Number of weights: ', len(weights))
    
    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            wt = torch.from_numpy(weights[ptr:ptr + nw])
            if len(wt) < nw:
                print('Elements less in weights = ', nw - len(wt))
                print('Filling with last seen element...')
                wt = torch.cat([wt, wt[-(nw - len(wt)):]], dim = 0)
                conv.weight.data.copy_(wt.view_as(conv.weight))
            else:
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw
        if mdef['type'] == 'Scale':
            cscale = module
            nb = cscale.beta.numel()
            nw = cscale.alpha.numel()
            print(nb, nw)
            cscale.beta.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(cscale.beta))
            ptr += nb
            cscale.alpha.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(cscale.alpha))
            ptr += nw
            
            
            


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
            
                
if __name__ == '__main__':
    # from torchviz import make_dot
    img_hw = (192, 640) 
    #img_hw = (368, 640)
    weights_path = './weights/best.pt'
    config_path = 'cfg/yolov3_custom.cfg'
    #weights_path = './weights/yolov3.pt'
    #config_path = 'cfg/yolov3.cfg'
    #weights_path = './weights/resnet10.pt'
    #config_path = 'cfg/resnet10.cfg'
    model = Darknet(config_path, img_hw, arch = 'yolov3', cls_act = 'sigmoid', onnx_flag = True, use_exp = False)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    #summary(model, (3, img_hw[0], img_hw[1]))    
    x = torch.zeros((1, 3,  img_hw[0], img_hw[1])).to(device)
    y = model(x)
    print(y[0].shape)
    
    #print(y[0].shape, y[1].shape)
    #make_dot(y[0]).render("attached", format="png")
    save_path = "../combined_model_benchmark/Onnx/yolov3.onnx"
    save_path = 'yolov3.onnx'
    
    
    input_names = ["image_input" ]
    #output_names = [ "conv2d_bbox", "conv2d_cov/Sigmoid"]
    
    print("Saving onnx file to {} ...".format(save_path))
    torch_out = torch.onnx._export(model,                   # model being run
                                   x,                       # model input (or a tuple for multiple inputs)
                                   save_path,               # where to save the model (can be a file or file-like object)
                                   export_params=True,      # store the trained parameter weights inside the model file
                                   do_constant_folding=True,
                                   # opset_version=11,
                                   input_names = input_names, 
                                   #output_names = output_names,
                                   keep_initializers_as_inputs = True
                                   )
