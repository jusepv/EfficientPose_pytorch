from functools import reduce

# import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torchvision.models as models
import torchvision
# from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
from efficientnet_pytorch import EfficientNet
from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization, RegressTranslation, CalculateTxTy, GroupNormalization
from initializers import PriorProbability
from utils.anchors import anchors_for_shape
import numpy as np



MOMENTUM = 0.997
EPSILON = 1e-4
EfficientNetB0 = EfficientNet.from_name('efficientnet-b0')
class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# my_view = lambda x, shape: x.view(shape)

def build_EfficientPose(phi,
                        num_classes=8,
                        num_anchors=9,
                        freeze_bn=False,
                        score_threshold=0.5,
                        anchor_parameters=None,
                        num_rotation_parameters=3,
                        print_architecture=True):
    assert phi in range(7)
    scaled_parameters = get_scaled_parameters(phi)

    input_size = scaled_parameters["input_size"]
    input_shape = (3, input_size, input_size)
    bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
    bifpn_depth = scaled_parameters["bifpn_depth"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters["num_groups_gn"]
    backbone_class = scaled_parameters["backbone_class"]

    # input layers
    image_input = nn.Parameter(torch.Tensor(*input_shape))
    camera_parameters_input = nn.Parameter(torch.Tensor(6))

    # build EfficientNet backbone
    backbone_feature_maps = backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)

    # build BiFPN
    fpn_feature_maps = build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, freeze_bn)

    # build subnets
    box_net, class_net, rotation_net, translation_net = build_subnets(num_classes,
                                                                      subnet_width,
                                                                      subnet_depth,
                                                                      subnet_num_iteration_steps,
                                                                      num_groups_gn,
                                                                      num_rotation_parameters,
                                                                      freeze_bn,
                                                                      num_anchors)

    # apply subnets to feature maps
    classification, bbox_regression, rotation, translation, transformation, bboxes = apply_subnets_to_feature_maps(
        box_net,
        class_net,
        rotation_net,
        translation_net,
        fpn_feature_maps,
        image_input,
        camera_parameters_input,
        input_size,
        anchor_parameters
    )

    # get the EfficientPose model for training without NMS and the rotation and translation output combined in the
    # transformation output because of the loss calculation
    efficientpose_train = nn.Sequential(
        box_net,
        class_net,
        rotation_net,
        translation_net
    )

    # filter detections (apply NMS / score threshold / select top-k)
    filtered_detections = FilterDetections(num_rotation_parameters=num_rotation_parameters,
                                           num_translation_parameters=3,
                                           score_threshold=score_threshold)([bboxes, classification, rotation, translation])

    efficientpose_prediction = nn.Sequential(filtered_detections)

    if print_architecture:
        print_models(efficientpose_train, box_net, class_net, rotation_net, translation_net)

    # create list with all layers to be able to load all layer weights because sometimes the whole subnet weight
    # loading is skipped if the output shape does not match instead of skipping just the output layer
    all_layers = list(set(efficientpose_train.modules()))

    return efficientpose_train, efficientpose_prediction, all_layers


def get_scaled_parameters(phi):
    """
    Get all needed scaled parameters to build EfficientPose
    Args:
        phi: EfficientPose scaling hyperparameter phi
    
    Returns:
       Dictionary containing the scaled parameters
    """
    #info tuples with scalable parameters
    image_sizes = (512)
    bifpn_widths = (64)
    bifpn_depths = (3)
    subnet_depths = (3)
    subnet_iteration_steps = (1)
    num_groups_gn = (4) #try to get 16 channels per group
    backbones = (EfficientNetB0,
                 )
    
    parameters = {"input_size": image_sizes[phi],
                  "bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi],
                  "backbone_class": backbones[phi]}
    
    return parameters


def build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, freeze_bn):
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    Args:
        backbone_feature_maps: Sequence containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layer
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       fpn_feature_maps: Sequence of BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    fpn_feature_maps = backbone_feature_maps
    for i in range(bifpn_depth):
        fpn_feature_maps = build_BiFPN_layer(fpn_feature_maps, bifpn_width, i, freeze_bn = freeze_bn)
        
    return fpn_feature_maps


def build_BiFPN_layer(features, num_channels, idx_BiFPN_layer, freeze_bn = False):
    """
    Builds a single layer of the bidirectional feature pyramid
    Args:
        features: Sequence containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    if idx_BiFPN_layer == 0:
        _, _, C3, C4, C5 = features
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    #top down pathway
    input_feature_maps_top_down = [P7_in,
                                   P6_in,
                                   P5_in_1 if idx_BiFPN_layer == 0 else P5_in,
                                   P4_in_1 if idx_BiFPN_layer == 0 else P4_in,
                                   P3_in]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer)
    
    #bottom up pathway
    input_feature_maps_bottom_up = [[P3_out],
                                    [P4_in_2 if idx_BiFPN_layer == 0 else P4_in, P4_td],
                                    [P5_in_2 if idx_BiFPN_layer == 0 else P5_in, P5_td],
                                    [P6_in, P6_td],
                                    [P7_in]]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer)
    
    
    return P3_out, P4_td, P5_td, P6_td, P7_out #TODO check if it is a bug to return the top down feature maps instead of the output maps


def prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn):
    """
    Prepares the backbone feature maps for the first BiFPN layer
    Args:
        C3, C4, C5: The EfficientNet backbone feature maps of the different levels
        num_channels: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The prepared input feature maps for the first BiFPN layer
    """
    P3_in = C3
    P3_in = nn.Conv2d(num_channels, kernel_size = 1, padding = 'same', name = 'fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d')(P3_in)
    P3_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode3/resample_0_0_8/bn')(P3_in)
    
    P4_in = C4
    P4_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d')(P4_in)
    P4_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode2/resample_0_1_7/bn')(P4_in_1)
    P4_in_2 = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d')(P4_in)
    P4_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode4/resample_0_1_9/bn')(P4_in_2)
    
    P5_in = C5
    P5_in_1 = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d')(P5_in)
    P5_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode1/resample_0_2_6/bn')(P5_in_1)
    P5_in_2 = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d')(P5_in)
    P5_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='fpn_cells/cell_0/fnode5/resample_0_2_10/bn')(P5_in_2)
    
    P6_in = nn.Conv2d(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
    P6_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
    P6_in = nn.MaxPool2d(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
    
    P7_in = nn.MaxPool2d(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in


def top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer):
    """
    Computes the top-down-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing the input feature maps of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the top-down-pathway
    """
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level = output_top_down_feature_maps[-1],
                                                    feature_maps_current_level = [input_feature_maps_top_down[level]],
                                                    upsampling = True,
                                                    num_channels = num_channels,
                                                    idx_BiFPN_layer = idx_BiFPN_layer,
                                                    node_idx = level - 1,
                                                    op_idx = 4 + level)
        
        output_top_down_feature_maps.append(merged_feature_map)
        
    return output_top_down_feature_maps


def bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the bottom-up-pathway
    """
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level = output_bottom_up_feature_maps[-1],
                                                    feature_maps_current_level = input_feature_maps_bottom_up[level],
                                                    upsampling = False,
                                                    num_channels = num_channels,
                                                    idx_BiFPN_layer = idx_BiFPN_layer,
                                                    node_idx = 3 + level,
                                                    op_idx = 8 + level)
        
        output_bottom_up_feature_maps.append(merged_feature_map)
        
    return output_bottom_up_feature_maps


def single_BiFPN_merge_step(feature_map_other_level, feature_maps_current_level, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx):
    """
    Merges two feature maps of different levels in the BiFPN
    Args:
        feature_map_other_level: Input feature map of a different level. Needs to be resized before merging.
        feature_maps_current_level: Input feature map of the current level
        upsampling: Boolean indicating wheter to upsample or downsample the feature map of the different level to match the shape of the current level
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        node_idx, op_idx: Integers needed to set the correct layer names
    
    Returns:
       The merged feature map
    """
    if upsampling:
        feature_map_resampled = F.interpolate(feature_map_other_level, scale_factor=2, mode='nearest')
    else:
        feature_map_resampled = nn.MaxPool2d(pool_size = 3, strides = 2, padding = 'same')(feature_map_other_level)
    
    merged_feature_map = wBiFPNAdd(name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/add')(feature_maps_current_level + [feature_map_resampled])
    # merged_feature_map = layers.Activation(lambda x: F.swish(x))(merged_feature_map)
    merged_feature_map = F.swish(merged_feature_map)
    merged_feature_map = SeparableConvBlock(num_channels = num_channels,
                                            kernel_size = 3,
                                            strides = 1,
                                            name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}')(merged_feature_map)

    return merged_feature_map


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn = False):
    """
    Builds a small block consisting of a depthwise separable convolution layer and a batch norm layer
    Args:
        num_channels: Number of channels used in the BiFPN
        kernel_size: Kernel site of the depthwise separable convolution layer
        strides: Stride of the depthwise separable convolution layer
        name: Name of the block
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The depthwise separable convolution block
    """
    f1 = SeparableConv2d(num_channels, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = True, name = f'{name}/conv')
    f2 = BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn, num_rotation_parameters, freeze_bn, num_anchors):
    """
    Builds the EfficientPose subnetworks
    Args:
        num_classes: Number of classes for the classification network output
        subnet_width: The number of channels used in the subnetwork layers
        subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    
    Returns:
       The subnetworks
    """
    box_net = BoxNet(subnet_width,
                      subnet_depth,
                      num_anchors = num_anchors,
                      freeze_bn = freeze_bn,
                      name = 'box_net')
    
    class_net = ClassNet(subnet_width,
                          subnet_depth,
                          num_classes = num_classes,
                          num_anchors = num_anchors,
                          freeze_bn = freeze_bn,
                          name = 'class_net')
    
    rotation_net = RotationNet(subnet_width,
                                subnet_depth,
                                num_values = num_rotation_parameters,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = True,
                                num_groups_gn = num_groups_gn,
                                name = 'rotation_net')
    
    translation_net = TranslationNet(subnet_width,
                                subnet_depth,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = True,
                                num_groups_gn = num_groups_gn,
                                name = 'translation_net')

    return box_net, class_net, rotation_net, translation_net     


class BoxNet(nn.Module):
    def __init__(self, width, depth, num_anchors = 9, freeze_bn = False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [SeparableConv2d(filters = self.width, name = f'{self.name}/box-{i}', **options) for i in range(self.depth)]
        self.head = SeparableConv2d(filters = self.num_anchors * self.num_values, name = f'{self.name}/box-predict', **options)
        
        self.bns = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.activation = F.swich()
        # self.reshape = layers.Reshape((-1, self.num_values))
        self.reshape = self.num_values.view(-1,)
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(nn.Module):
    def __init__(self, width, depth, num_classes = 8, num_anchors = 9, freeze_bn = False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [SeparableConv2d(filters = self.width, bias_initializer = 'zeros', name = f'{self.name}/class-{i}', **options) for i in range(self.depth)]
        self.head = SeparableConv2d(filters = self.num_classes * self.num_anchors, bias_initializer = PriorProbability(probability = 0.01), name = f'{self.name}/class-predict', **options)

        self.bns = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        # self.reshape = layers.Reshape((-1, self.num_classes))
        self.activation = F.swich()
        self.reshape = self.num_values.view(-1,)
        self.activation_sigmoid = F.relu()
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation_sigmoid(outputs)
        self.level += 1
        return outputs
    
    
class IterativeRotationSubNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, **kwargs):
        super(IterativeRotationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        # if backend.image_data_format() == 'channels_first':
        #     gn_channel_axis = 1
        # else:
        #     gn_channel_axis = -1
        gn_channel_axis = 3
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [SeparableConv2d(filters = width, name = f'{self.name}/iterative-rotation-sub-{i}', **options) for i in range(self.depth)]
        self.head = SeparableConv2d(filters = self.num_anchors * self.num_values, name = f'{self.name}/iterative-rotation-sub-predict', **options)
        
        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/iterative-rotation-sub-{k}-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/iterative-rotation-sub-{k}-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.activation = F.swish()

    def call(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        
        return outputs
    
    
class RotationNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, **kwargs):
        super(RotationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/rotation-{i}', **options) for i in range(self.depth)]
        self.initial_rotation = layers.SeparableConv2D(filters = self.num_anchors * self.num_values, name = f'{self.name}/rotation-init-predict', **options)
    
        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/rotation-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/rotation-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeRotationSubNet(width = self.width,
                                                          depth = self.depth - 1,
                                                          num_values = self.num_values,
                                                          num_iteration_steps = self.num_iteration_steps,
                                                          num_anchors = self.num_anchors,
                                                          freeze_bn = freeze_bn,
                                                          use_group_norm = self.use_group_norm,
                                                          num_groups_gn = self.num_groups_gn,
                                                          name = "iterative_rotation_subnet")

        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.activatoin = F.swich()
        self.reshape = layers.Reshape((-1, num_values))
        self.reshape = my_view()
        self.level = 0
        self.add = torch.add()
        self.concat = torch.cat(axis = channel_axis)

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)
            
        rotation = self.initial_rotation(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, rotation])
            delta_rotation = self.iterative_submodel([iterative_input, level], level_py = self.level, iter_step_py = i)
            rotation = self.add([rotation, delta_rotation])
        
        outputs = self.reshape(rotation)
        self.level += 1
        return outputs
    
    
class IterativeTranslationSubNet(nn.Module):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, **kwargs):
        super(IterativeTranslationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [SeparableConv2d(filters = self.width, name = f'{self.name}/iterative-translation-sub-{i}', **options) for i in range(self.depth)]
        self.head_xy = SeparableConv2d(filters = self.num_anchors * 2, name = f'{self.name}/iterative-translation-xy-sub-predict', **options)
        self.head_z = SeparableConv2d(filters = self.num_anchors, name = f'{self.name}/iterative-translation-z-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/iterative-translation-sub-{k}-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[BatchNormalization(freeze = freeze_bn, momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/iterative-translation-sub-{k}-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.activation = F.swish()


    def call(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)

        return outputs_xy, outputs_z
    
    
    
class TranslationNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, **kwargs):
        super(TranslationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
            'pointwise_initializer': init.kaiming_uniform_(a=0, mode='fan_in', nonlinearity='linear'),
        }
        options.update(kernel_initializer)
        self.convs = [SeparableConv2d(filters = self.width, name = f'{self.name}/translation-{i}', **options) for i in range(self.depth)]
        self.initial_translation_xy = SeparableConv2d(filters = self.num_anchors * 2, name = f'{self.name}/translation-xy-init-predict', **options)
        self.initial_translation_z = SeparableConv2d(filters = self.num_anchors, name = f'{self.name}/translation-z-init-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups = self.num_groups_gn, axis = gn_channel_axis, name = f'{self.name}/translation-{i}-gn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/translation-{i}-bn-{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeTranslationSubNet(width = self.width,
                                                             depth = self.depth - 1,
                                                             num_iteration_steps = self.num_iteration_steps,
                                                             num_anchors = self.num_anchors,
                                                             freeze_bn = freeze_bn,
                                                             use_group_norm= self.use_group_norm,
                                                             num_groups_gn = self.num_groups_gn,
                                                             name = "iterative_translation_subnet")

        # self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.activation = F.swish()
        # self.reshape_xy = layers.Reshape((-1, 2))
        self.reshape_xy = torch.reshape((-1,2))
        self.reshape_z = torch.reshape((-1, 1))
        self.level = 0
        self.add = torch.add()
        self.concat = torch.cat(axis = channel_axis)
            
        self.concat_output = torch.cat(axis = -1) #always last axis after reshape

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            # feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)
            
        translation_xy = self.initial_translation_xy(feature)
        translation_z = self.initial_translation_z(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, translation_xy, translation_z])
            delta_translation_xy, delta_translation_z = self.iterative_submodel([iterative_input, level], level_py = self.level, iter_step_py = i)
            translation_xy = self.add([translation_xy, delta_translation_xy])
            translation_z = self.add([translation_z, delta_translation_z])
        # import pdb;pdb.set_trace()

        outputs_xy = self.reshape_xy(translation_xy)
        outputs_z = self.reshape_z(translation_z)
        outputs = self.concat_output([outputs_xy, outputs_z])
        self.level += 1
        return outputs
    

def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, image_input, camera_parameters_input, input_size, anchor_parameters):
    """
    Applies the subnetworks to the BiFPN feature maps
    Args:
        box_net, class_net, rotation_net, translation_net: Subnetworks
        fpn_feature_maps: Sequence of the BiFPN feature maps of the different levels (P3, P4, P5, P6, P7)
        image_input, camera_parameters_input: The image and camera parameter input layer
        input size: Integer representing the input image resolution
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
    
    Returns:
       classification: Tensor containing the classification outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_classes)
       bbox_regression: Tensor containing the deltas of anchor boxes to the GT 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
       rotation: Tensor containing the rotation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters)
       translation: Tensor containing the translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, 3)
       transformation: Tensor containing the concatenated rotation and translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters + 3)
                       Rotation and Translation are concatenated because the Keras Loss function takes only one GT and prediction tensor respectively as input but the transformation loss needs both
       bboxes: Tensor containing the 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
    """


    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    classification = torch.cat(axis=1, name='classification')(classification)

    bbox_regression = [box_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    bbox_regression = torch.cat(axis=1, name='regression')(bbox_regression)
    
    rotation = [rotation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    rotation = torch.cat(axis = 1, name='rotation')(rotation)
    
    translation_raw = [translation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    translation_raw = torch.cat(axis = 1, name='translation_raw_outputs')(translation_raw)
    
    #get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params = anchor_parameters)
    translation_anchors_input = np.expand_dims(translation_anchors, axis = 0)
    
    translation_xy_Tz = RegressTranslation(name = 'translation_regression')([translation_anchors_input, translation_raw])
    translation = CalculateTxTy(name = 'translation')(translation_xy_Tz,
                                                        fx = camera_parameters_input[:, 0],
                                                        fy = camera_parameters_input[:, 1],
                                                        px = camera_parameters_input[:, 2],
                                                        py = camera_parameters_input[:, 3],
                                                        tz_scale = camera_parameters_input[:, 4],
                                                        image_scale = camera_parameters_input[:, 5])
    
    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis = 0)
    bboxes = RegressBoxes(name='boxes')([anchors_input, bbox_regression[..., :4]])
    bboxes = ClipBoxes(name='clipped_boxes')([image_input, bboxes])
    
    #concat rotation and translation outputs to transformation output to have a single output for transformation loss calculation
    #standard concatenate layer throws error that shapes does not match because translation shape dim 2 is known via translation_anchors and rotation shape dim 2 is None
    #so just use lambda layer with tf concat
    # transformation = layers.Lambda(lambda input_list: tf.concat(input_list, axis = -1), name="transformation")([rotation, translation])
    transformation = torch.cat([rotation, translation], dim=-1)

    return classification, bbox_regression, rotation, translation, transformation, bboxes
    

def print_models(*models):
    """
    Print the model architectures
    Args:
        *models: Tuple containing all models that should be printed
    """
    for model in models:
        print("\n\n")
        # model.summary()
        print("\n\n")
