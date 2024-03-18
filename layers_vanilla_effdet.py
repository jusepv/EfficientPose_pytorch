"""
Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn

class BatchNormalization(nn.Module):
    def __init__(self, freeze, *args, **kwargs):
        super(BatchNormalization, self).__init__()
        self.freeze = freeze
        self.bn = nn.BatchNorm2d(*args, **kwargs)
        if freeze:
            self.bn.eval()
            for param in self.bn.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        if self.training:
            return self.bn(inputs)
        else:
            return self.bn(inputs)

    def freeze_bn(self):
        self.bn.eval()
        for param in self.bn.parameters():
            param.requires_grad = False

    def unfreeze_bn(self):
        self.bn.train()
        for param in self.bn.parameters():
            param.requires_grad = True


class wBiFPNAdd(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(wBiFPNAdd, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        w = torch.relu(inputs)
        w_sum = torch.sum(w, dim=0)
        weighted_sum = torch.sum(inputs * w, dim=0)
        x = weighted_sum / (w_sum + self.epsilon)
        return x



def bbox_transform_inv(boxes, deltas, scale_factors=None):
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


class ClipBoxes(torch.nn.Module):
    def forward(self, inputs):
        image, boxes = inputs
        shape = image.shape
        height = shape[1]
        width = shape[2]
        x1 = torch.clamp(boxes[:, :, 0], 0, width - 1)
        y1 = torch.clamp(boxes[:, :, 1], 0, height - 1)
        x2 = torch.clamp(boxes[:, :, 2], 0, width - 1)
        y2 = torch.clamp(boxes[:, :, 3], 0, height - 1)

        return torch.stack([x1, y1, x2, y2], dim=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class RegressBoxes(torch.nn.Module):
    def forward(self, inputs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config



def filter_detections(
        boxes,
        classification,
        alphas=None,
        ratios=None,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.01,
        max_detections=100,
        nms_threshold=0.5,
        detect_quadrangle=False,
):
    def _filter_detections(scores_, labels_):
        indices_ = torch.where(scores_ > score_threshold)

        if nms:
            filtered_boxes = boxes[indices_]
            filtered_scores = scores_[indices_]

            # perform NMS
            nms_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices_ = indices_[nms_indices]

        labels_ = labels_[indices_]
        indices_ = torch.stack([indices_[:, 0], labels_], dim=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(classification.shape[1]):
            scores = classification[:, c]
            labels = c * torch.ones_like(scores, dtype=torch.int64)
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = torch.cat(all_indices, dim=0)
    else:
        scores = classification.max(dim=1).values
        labels = classification.argmax(dim=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores, top_indices = torch.topk(scores, k=min(max_detections, scores.shape[0]))

    # filter input using the final set of indices
    indices = indices[top_indices]
    boxes = boxes[indices[:, 0]]
    labels = labels[top_indices]

    # zero pad the outputs
    pad_size = max(0, max_detections - scores.shape[0])
    boxes = torch.nn.functional.pad(boxes, [0, 0, 0, pad_size], value=-1)
    scores = torch.nn.functional.pad(scores, [0, pad_size], value=-1)
    labels = torch.nn.functional.pad(labels, [0, pad_size], value=-1)
    labels = labels.int()

    # set shapes, since we know what they are
    boxes = boxes.view(max_detections, 4)
    scores = scores.view(max_detections)

    if detect_quadrangle:
        alphas = alphas[indices[:, 0]]
        ratios = ratios[indices[:, 0]]
        alphas = torch.nn.functional.pad(alphas, [0, 0, 0, pad_size], value=-1)
        ratios = torch.nn.functional.pad(ratios, [0, pad_size], value=-1)
        alphas = alphas.view(max_detections, 4)
        ratios = ratios.view(max_detections)
        return [boxes, scores, alphas, ratios, labels]
    else:
        return [boxes, scores, labels]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.01,
            max_detections=100,
            parallel_iterations=32,
            detect_quadrangle=False,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.detect_quadrangle = detect_quadrangle
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        if self.detect_quadrangle:
            alphas = inputs[2]
            ratios = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            alphas_ = args[2] if self.detect_quadrangle else None
            ratios_ = args[3] if self.detect_quadrangle else None

            return filter_detections(
                boxes_,
                classification_,
                alphas_,
                ratios_,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
                detect_quadrangle=self.detect_quadrangle,
            )

        # call filter_detections on each batch item
        if self.detect_quadrangle:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification, alphas, ratios],
                dtype=['float32', 'float32', 'float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )
        else:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification],
                dtype=['float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        if self.detect_quadrangle:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ]
        else:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config

