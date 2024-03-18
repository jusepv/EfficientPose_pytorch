# """
# Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
# """
# """
# Copyright 2017-2018 Fizyr (https://fizyr.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """

# # import keras
# from tensorflow import keras

# import numpy as np
# import math


# class PriorProbability(keras.initializers.Initializer):
#     """ Apply a prior probability to the weights.
#     """

#     def __init__(self, probability=0.01):
#         self.probability = probability

#     def get_config(self):
#         return {
#             'probability': self.probability
#         }

#     def __call__(self, shape, dtype=None):
#         # set bias to -log((1 - p)/p) for foreground
#         result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

#         return result


import torch
import numpy as np
import math

class PriorProbability(torch.nn.Module):
    def __init__(self, probability=0.01):
        super(PriorProbability, self).__init__()
        self.probability = probability

    def forward(self, input):
        result = torch.ones_like(input) * -torch.log((1 - self.probability) / self.probability)
        return result