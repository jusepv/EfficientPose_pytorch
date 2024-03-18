"""
Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

EfficientNetB0 = EfficientNet.from_pretrained('efficientnet-b0')
EfficientNetB1 = EfficientNet.from_pretrained('efficientnet-b1')
EfficientNetB2 = EfficientNet.from_pretrained('efficientnet-b2')
EfficientNetB3 = EfficientNet.from_pretrained('efficientnet-b3')
EfficientNetB4 = EfficientNet.from_pretrained('efficientnet-b4')
EfficientNetB5 = EfficientNet.from_pretrained('efficientnet-b5')
EfficientNetB6 = EfficientNet.from_pretrained('efficientnet-b6')
EfficientNetB7 = EfficientNet.from_pretrained('efficientnet-b7')

preprocess_input = EfficientNet.preprocess_input