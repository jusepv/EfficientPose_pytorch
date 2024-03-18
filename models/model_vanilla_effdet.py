    """
    Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
    """

    from functools import reduce
    import torch
    import torch.nn as nn
    import tensorflow as tf

    from torch_ed import EfficientNetB0, EfficientNetB1, EfficientNetB2
    from torch_ed import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

    from layers_vanilla_effdet import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
    from initializers import PriorProbability
    from utils.anchors import anchors_for_shape
    import numpy as np

    w_bifpns = [64, 88, 112, 160, 224, 288, 384]
    d_bifpns = [3, 4, 5, 6, 7, 7, 8]
    d_heads = [3, 3, 3, 4, 4, 4, 5]
    num_groups_gn = [4, 4, 7, 10, 14, 18, 24] #try to get 16 channels per group
    #d_iteratives = [2, 2, 2, 3, 3, 3, 4]
    iteration_steps = [1, 1, 1, 2, 2, 2, 3]
    image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
    backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
                EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]

    MOMENTUM = 0.997
    EPSILON = 1e-4


    class SeparableConvBlock(nn.Module):
        def __init__(self, num_channels, kernel_size, strides, name, freeze_bn=False):
            super(SeparableConvBlock, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=strides, padding='same'),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )

        def forward(self, x):
            return self.conv(x)

    class ConvBlock(nn.Module):
        def __init__(self, num_channels, kernel_size, strides, name, freeze_bn=False):
            super(ConvBlock, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=strides, padding='same'),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )

        def forward(self, x):
            return self.conv(x)


    def build_wBiFPN(features, num_channels, id, freeze_bn=False):
        if id == 0:
            _, _, C3, C4, C5 = features
            P3_in = C3
            P4_in = C4
            P5_in = C5
            P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
            # P6_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
            P6_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
            P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
            P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
            P7_U = layers.UpSampling2D()(P7_in)
            P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
            P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
            # P5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P5_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P6_U = layers.UpSampling2D()(P6_td)
            P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
            P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
            # P4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            P4_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            P5_U = layers.UpSampling2D()(P5_td)
            P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
            P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
            # P3_in = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                   name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
            P3_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
            P4_U = layers.UpSampling2D()(P4_td)
            P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
            P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
            # P4_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
            P4_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
            P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
            P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

            P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
            # P5_in_2 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
            P5_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
            P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
            P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

            P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
            P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

            P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
            P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
            P7_U = layers.UpSampling2D()(P7_in)
            P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
            P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P6_U = layers.UpSampling2D()(P6_td)
            P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
            P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P5_U = layers.UpSampling2D()(P5_td)
            P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
            P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P4_U = layers.UpSampling2D()(P4_td)
            P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
            P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
            P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

            P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
            P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

            P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
            P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

            P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
            P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                        name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
        return P3_out, P4_td, P5_td, P6_td, P7_out


    def build_BiFPN(features, num_channels, id, freeze_bn=False):
        if id == 0:
            _, _, C3, C4, C5 = features
            P3_in = C3
            P4_in = C4
            P5_in = C5
            # layers.Conv2D -> nn.Conv2d
            # BatchNormalization -> nn.BatchNorm2d
            # layers.MaxPooling2D -> nn.MaxPool2d
            # layers.UpSampling2D() -> nn.Upsample
            # layers.Add -> +
            # layers.Activation(lambda x: tf.nn.swish(x))(P6_td) -> nn.ReLU()
            P6_in = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
            P6_in = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON, name='resample_p6/bn')(P6_in)
            P6_in = nn.MaxPool2d(kernel_size=3, stride=2, padding='same', name='resample_p6/maxpool')(P6_in)
            P7_in = nn.MaxPool2d(kernel_size=3, stride=2, padding='same', name='resample_p7/maxpool')(P6_in)
            P7_U = nn.Upsample(scale_factor=2, mode='nearest')(P7_in)
            P6_td = P6_in + P7_U
            P6_td = nn.GELU()(P6_td)
            P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P5_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
            P5_in_1 = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P6_U = nn.Upsample(scale_factor=2, mode='nearest')(P6_td)
            P5_td = P5_in_1 + P6_U
            P5_td = nn.GELU()(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P4_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
            # P5_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P5_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P6_U = layers.UpSampling2D()(P6_td)
            P5_td = layers.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
            P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                    name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)


                                    
            # P4_in_1 = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
            #                                     name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            P6_in = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
            P6_in = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON, name='resample_p6/bn')(P6_in)
            P6_in = nn.MaxPool2d(kernel_size=3, stride=2, padding='same', name='resample_p6/maxpool')(P6_in)
            P7_in = nn.MaxPool2d(kernel_size=3, stride=2, padding='same', name='resample_p7/maxpool')(P6_in)
            P7_U = nn.Upsample(scale_factor=2, mode='nearest')(P7_in)
            P6_td = P6_in + P7_U
            P6_td = nn.GELU()(P6_td)
            P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                    name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P5_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
            P5_in_1 = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
            P6_U = nn.Upsample(scale_factor=2, mode='nearest')(P6_td)
            P5_td = P5_in_1 + P6_U
            P5_td = nn.GELU()(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                    name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P4_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
            P4_in_1 = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
            P5_U = nn.Upsample(scale_factor=2, mode='nearest')(P5_td)
            P4_td = P4_in_1 + P5_U
            P4_td = nn.GELU()(P4_td)
            P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                    name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P3_in = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                            name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
            P3_in = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
            P4_U = nn.Upsample(scale_factor=2, mode='nearest')(P4_td)
            P3_out = P3_in + P4_U
            P3_out = nn.GELU()(P3_out)
            P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P4_in_2 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
            P4_in_2 = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
            P3_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P3_out)
            P4_out = P4_in_2 + P4_td + P3_D
            P4_out = nn.GELU()(P4_out)
            P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)
            P5_in_2 = nn.Conv2d(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
            P5_in_2 = nn.BatchNorm2d(num_channels, momentum=MOMENTUM, eps=EPSILON,
                                    name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
            P4_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P4_out)
            P5_out = P5_in_2 + P5_td + P4_D
            P5_out = nn.GELU()(P5_out)
            P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)
            P5_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P5_out)
            P6_out = P6_in + P6_td + P5_D
            P6_out = nn.GELU()(P6_out)
            P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)
            P6_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P6_out)
            P7_out = P7_in + P6_D
            P7_out = nn.GELU()(P7_out)

        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
            P7_U = nn.Upsample(scale_factor=2, mode='nearest')(P7_in)
            P6_td = P6_in + P7_U
            P6_td = nn.ReLU()(P6_td)
            P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
            P6_U = nn.Upsample(scale_factor=2, mode='nearest')(P6_td)
            P5_td = P5_in + P6_U
            P5_td = nn.ReLU()(P5_td)
            P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
            P5_U = nn.Upsample(scale_factor=2, mode='nearest')(P5_td)
            P4_td = P4_in + P5_U
            P4_td = nn.ReLU()(P4_td)
            P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
            P4_U = nn.Upsample(scale_factor=2, mode='nearest')(P4_td)
            P3_out = P3_in + P4_U
            P3_out = nn.ReLU()(P3_out)
            P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
            P3_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P3_out)
            P4_out = P4_in + P4_td + P3_D
            P4_out = nn.ReLU()(P4_out)
            P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)
            P4_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P4_out)
            P5_out = P5_in + P5_td + P4_D
            P5_out = nn.ReLU()(P5_out)
            P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)
            P5_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P5_out)
            P6_out = P6_in + P6_td + P5_D
            P6_out = nn.ReLU()(P6_out)
            P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)
            P6_D = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')(P6_out)
            P7_out = P7_in + P6_D
            P7_out = nn.ReLU()(P7_out)
            P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, stride=1,
                                        name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

        return P3_out, P4_td, P5_td, P6_td, P7_out


    class BoxNet(nn.Module):
        def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
            super(BoxNet, self).__init__(**kwargs)
            self.width = width
            self.depth = depth
            self.num_anchors = num_anchors
            self.separable_conv = separable_conv
            self.detect_quadrangle = detect_quadrangle
            num_values = 9 if detect_quadrangle else 4
            options = {
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'bias': True,
            }
            if separable_conv:
                kernel_initializer = {
                    'depthwise_initializer': torch.nn.init.xavier_uniform_,
                    'pointwise_initializer': torch.nn.init.xavier_uniform_,
                }
                options.update(kernel_initializer)
                self.convs = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(width, width, **options),
                    nn.BatchNorm2d(width),
                    nn.ReLU()
                ) for _ in range(depth)])
                self.head = nn.Sequential(
                    nn.Conv2d(width, num_anchors * num_values, **options),
                    nn.BatchNorm2d(num_anchors * num_values),
                    nn.ReLU()
                )
            else:
                kernel_initializer = {
                    'kernel_initializer': torch.nn.init.normal_
                }
                options.update(kernel_initializer)
                self.convs = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(width, width, **options),
                    nn.BatchNorm2d(width),
                    nn.ReLU()
                ) for _ in range(depth)])
                self.head = nn.Sequential(
                    nn.Conv2d(width, num_anchors * num_values, **options),
                    nn.BatchNorm2d(num_anchors * num_values),
                    nn.ReLU()
                )
            self.bns = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(3, 8)]) for _ in range(depth)])
            self.relu = nn.SiLU()
            self.reshape = nn.Flatten(start_dim=1)
            self.level = 0

        def forward(self, inputs, **kwargs):
            feature, level = inputs
            for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[i][self.level - 1](feature)
                feature = self.relu(feature)
            outputs = self.head(feature)
            outputs = self.reshape(outputs)
            self.level += 1
            return outputs


class ClassNet(nn.Module):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': torch.nn.init.xavier_uniform_,
                'pointwise_initializer': torch.nn.init.xavier_uniform_,
            }
            options.update(kernel_initializer)
            self.convs = nn.ModuleList([nn.Sequential(
                nn.Conv2d(width, width, **options),
                nn.BatchNorm2d(width),
                nn.ReLU()
            ) for _ in range(depth)])
            self.head = nn.Sequential(
                nn.Conv2d(width, num_classes * num_anchors, **options),
                nn.BatchNorm2d(num_classes * num_anchors),
                nn.ReLU()
            )
        else:
            kernel_initializer = {
                'kernel_initializer': torch.nn.init.normal_
            }
            options.update(kernel_initializer)
            self.convs = nn.ModuleList([nn.Sequential(
                nn.Conv2d(width, width, **options),
                nn.BatchNorm2d(width),
                nn.ReLU()
            ) for _ in range(depth)])
            self.head = nn.Sequential(
                nn.Conv2d(width, num_classes * num_anchors, **options),
                nn.BatchNorm2d(num_classes * num_anchors),
                nn.ReLU()
            )
        self.bns = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(width) for _ in range(3, 8)]) for _ in range(depth)])
        self.relu = nn.SiLU()
        self.reshape = nn.Flatten(start_dim=1)
        self.activation = nn.Sigmoid()
        self.level = 0

    def forward(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level - 1](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        return outputs
        


def efficientdet(phi, num_classes=20, num_anchors=9, weighted_bifpn=False, freeze_bn=False,
                score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True, num_rotation_parameters=3):
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (3, input_size, input_size)
    image_input = torch.nn.functional.interpolate(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi]
    w_head = w_bifpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    if weighted_bifpn:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_BiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)
    box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, separable_conv=separable_conv, freeze_bn=freeze_bn,
                    detect_quadrangle=detect_quadrangle, name='box_net')
    class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors,
                        separable_conv=separable_conv, freeze_bn=freeze_bn, name='class_net')

    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
    classification = torch.cat(classification, dim=1)

    regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
    regression = torch.cat(regression, dim=1)

    # get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)

    # print subnets
    print("\n\nBox Net\n")
    print(box_net)

    print("\n\nClass Net\n")
    print(class_net)

    # model = models.Model(inputs=[image_input], outputs=[classification, regression, rotation, translation], name='efficientdet')
    model = nn.Sequential(image_input, [classification, regression], name='efficientdet')

    # create list with all layers to be able to load all layer weights
    all_layers = list(set(model.layers + box_net.layers + class_net.layers))

    # apply predicted regression to anchors
    anchors_input = anchors.unsqueeze(0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    if detect_quadrangle:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold,
            detect_quadrangle=True
        )([boxes, classification, regression[..., 4:8], regression[..., 8]])
    else:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold
        )([boxes, classification])

    prediction_model = nn.Sequential(image_input, detections, name='efficientdet_p')
    return model


    if __name__ == '__main__':
        x, y = efficientdet(1)
