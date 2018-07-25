"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from . import backend


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels         = y_true
        classification = y_pred

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # filter out "ignore" anchors
        anchor_state = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices      = backend.where(keras.backend.not_equal(anchor_state, -1))
        cls_loss     = backend.gather_nd(cls_loss, indices)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # filter out "ignore" anchors
        indices         = backend.where(keras.backend.equal(anchor_state, 1))
        regression_loss = backend.gather_nd(regression_loss, indices)

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(keras.backend.maximum(1, normalizer), dtype=keras.backend.floatx())

        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


from keras.engine.topology import Layer


def discrepancy_clas(inputs1, inputs2):
    ''' inputs1, inputs2 are tensors of shape (?, ?, num_classes) '''
    label_axis = 2
    #inputs1 = keras.backend.softmax(inputs1) # FIXME when have TF1.6:   , axis=label_axis)
    #inputs2 = keras.backend.softmax(inputs2) # FIXME same:  , axis=label_axis)
    diff = keras.backend.abs(inputs1 - inputs2)
    print ('keras.backend.mean(diff)', keras.backend.mean(diff).get_shape())
    return keras.backend.mean(diff)

class DiscrepancyClas(Layer):
    def __init__(self, **kwargs):
        super(DiscrepancyClas, self).__init__(**kwargs)

    def call(self, x, mask=None):
        loss = discrepancy_clas(x[0], x[1])
        self.add_loss(loss, x)
        return loss

    def get_output_shape(self, input_shape):
        #return (input_shape[0][0],1)  # TODO: confirm
        return (1,)

 
def discrepancy_regr(inputs1, inputs2):
    return keras.backend.mean(keras.backend.abs(inputs1 - inputs2))

class DiscrepancyRegr(Layer):
    def __init__(self, **kwargs):
        super(DiscrepancyRegr, self).__init__(**kwargs)

    def call(self, x, mask=None):
        loss = discrepancy_regr(x[0], x[1])
        self.add_loss(loss, x)
        return loss

    def get_output_shape(self, input_shape):
        #return (input_shape[0][0],1)  # TODO: confirm
        return (1,)


def zero_loss(y_true, y_pred):
    return keras.backend.zeros_like(y_pred)  # Need to fix shape.

