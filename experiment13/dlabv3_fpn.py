# import math
# import fvcore.nn.weight_init as weight_init
# import torch
# import torch.nn.functional as F
# from torch import nn

import torch
from torch import nn
from torch.nn import functional as F

# from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.layers import ShapeSpec

from detectron2.modeling.backbone.backbone import Backbone

import torchvision.models.segmentation as seg_models
import torchvision.models as models

import copy

__all__ = ["DeepLabV3plus_FPN"]



class DeepLabV3plus_FPN(Backbone):
    def __init__(self, out_channels, out_features, strides):
        super().__init__()
        self.backbone = seg_models.deeplabv3_resnet101(pretrained=True, progress=True).backbone
        inplanes1 = 1024
        inplanes2 = 2048
        low_level_planes = 512 # was 256

        aspp1_dilate = [12, 24, 36] # for layer 4
        aspp2_dilate = [ 6, 12, 18] # for layer 3
        self.classifier = DeepLabHeadV3Plus(
            inplanes1,
            inplanes2,
            low_level_planes,
            num_classes=21,
            aspp1_dilate=aspp1_dilate,
            aspp2_dilate=aspp2_dilate,
        )
        self._out_features = out_features
        self._out_channels = out_channels
        self._out_feature_strides = strides


        self.novel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes2, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
        )


        self.conv1 = nn.Conv2d(48, 256, 1)
        self.conv2 = nn.Conv2d(21, 256, 1)

        convert_to_separable_conv(self.classifier)

        strides = [4, 8, 16, 32]
        self._size_divisibility = strides[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        results = {f: None for f in self._out_features}
        # ['p2', 'p3', 'p4', 'p5', 'p6']

        aux = None
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        aux = x
        x = self.backbone.layer3(x)
        x1 = x
        x2 = self.backbone.layer4(x)

        # classifier
        low_level = self.classifier.project(aux)
        results['p2'] = self.conv1(low_level)
        
        output_feature1 = self.classifier.aspp1(x1)
        output_feature2 = self.classifier.aspp2(x2)
        results['p3'] = output_feature1
        results['p4'] = output_feature2
        output_feature = output_feature2

        output_feature = F.interpolate(output_feature, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        concated = torch.cat([low_level, output_feature], dim=1)

        results['p5'] = self.conv2(self.classifier.classifier(concated))
        results['p6'] = self.novel(x2)

        return results


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_channels, stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }




class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels1, in_channels2, low_level_channels, num_classes, aspp1_dilate=[12, 24, 36], aspp2_dilate=[6, 12, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp1 = ASPP(in_channels1, aspp1_dilate)
        self.aspp2 = ASPP(in_channels2, aspp2_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 304, 3, padding=1, bias=False),
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        pass
        # NOT USING THE FORWARD FUNCTION    
        # low_level_feature = self.project(feature['aux'])
        # output_feature = self.aspp(feature['out'])
        # output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        # return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)












# class DeepLabV3_FPN(Backbone):

#     def __init__(self, out_channels, out_features, strides):
#         super().__init__()
#         self.dlabv3 = seg_models.deeplabv3_resnet101(pretrained=True, progress=True)
#         self._out_features = out_features
#         self._out_channels = out_channels
#         self._out_feature_strides = strides

#         strides = [4, 8, 16, 32]
#         self._size_divisibility = strides[-1]

#     @property
#     def size_divisibility(self):
#         return self._size_divisibility

#     def forward(self, x):
#         results = []
#         # backbone (down-scaling)
#         x = self.dlabv3.backbone(x)
#         print(x['out'].shape, x['aux'].shape)
#         x = x['out']

#         # classifier (up-scaling)
#         x = self.dlabv3.classifier[0].convs[0](x) # Conv2d > BatchNorm2d > ReLU
#         results.append(x) # --> for p6
#         x = self.dlabv3.classifier[0].convs[1](x) # ASPPConv
#         results.insert(0, x) # --> for p5
#         x = self.dlabv3.classifier[0].convs[2](x) # ASPPConv
#         results.insert(0, x) # --> for p4
#         x = self.dlabv3.classifier[0].convs[3](x) # ASPPConv
#         results.insert(0, x) # --> for p3
#         x = self.dlabv3.classifier[0].convs[4](x) # AdaptiveAvgPool2d > Conv2d > BatchNorm2d > ReLU
#         results.insert(0, x) # --> for p2
#         x = self.dlabv3.classifier[0].project(x)
#         x = self.dlabv3.classifier[1](x)
#         x = self.dlabv3.classifier[2](x)
#         x = self.dlabv3.classifier[3](x)
#         x = self.dlabv3.classifier[4](x)

#         # aux_classifier
#         x = self.dlabv3.aux_classifier(x)

#         assert len(self._out_features) == len(results)
#         return {f: res for f, res in zip(self._out_features, results)}

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_channels, stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }