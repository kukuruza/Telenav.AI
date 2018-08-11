"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import numpy as np
import logging
import os.path
from collections import defaultdict
from glob import glob

import apollo_python_common.image
import apollo_python_common.proto_api as meta
import apollo_python_common.io_utils as io_utils

import numpy as np
import random
import threading
import time
import warnings

import keras

from keras_retinanet.utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from keras_retinanet.utils.transform import transform_aabb


class ImageOnlyGenerator(object):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
    ):
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('size method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image):
        # randomly transform both image
        if self.transform_generator:
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

        return image

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group_entry(self, image):
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image
        image = self.random_transform_group_entry(image)

        # resize image
        image, image_scale = self.resize_image(image)

        return image

    def preprocess_group(self, image_group):
        for index, image in enumerate(image_group):
            # preprocess a single group entry
            image = self.preprocess_group_entry(image)

            # copy processed data back to group
            image_group[index]       = image

        return image_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_input_output(self, group):
        # load images
        image_group = self.load_image_group(group)

        # perform preprocessing steps
        image_group = self.preprocess_group(image_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        return inputs

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


class ImageFolderGenerator(ImageOnlyGenerator):
    def __init__(
            self,
            base_dir,
            transform_generator,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing TrafficSigns dataset from'.format(base_dir))
        self.image_names = []
        self.base_dir = base_dir
        self.image_names = glob(os.path.join(self.base_dir, '*.jpg'))
        super().__init__(transform_generator, **kwargs)
        self.logger.info('Dataset was initialised.')

    def size(self):
        return len(self.image_names)

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        return apollo_python_common.image.get_aspect_ratio(self.image_path(image_index))

    def image_size(self, image_index):
        return apollo_python_common.image.get_size(self.image_path(image_index))

    def load_image(self, image_index):
        #self.logger.info(self.image_path(image_index))
        img = apollo_python_common.image.get_bgr(self.image_path(image_index))
        return img


class AdaptGenerator:
    def __init__(self, generator_src, generator_tgt):
        self.generator_src = generator_src
        self.generator_tgt = generator_tgt
        self.logger = logging.getLogger(__name__)
        self.logger.info('Dataset was initialised.')

    def __next__(self):
        input_src, target_src, labels_src = self.generator_src.__next__()
        input_dst = self.generator_tgt.__next__()
        self.logger.debug('next input_src[0] shape: %s' % str(input_src[0].shape))
        self.logger.debug('next input_dst[0] shape: %s' % str(input_dst[0].shape))
        self.logger.debug('next target_src shape: %s' % str(len(target_src)))
        self.logger.debug('next target_src shape: %s' % str(np.asarray(target_src[0]).shape))
        self.logger.debug('next target_src shape: %s' % str(np.asarray(target_src[1]).shape))
        return {'src': input_src, 'dst': input_dst}, {'src': target_src}, {'src': labels_src}
        #return [input_src, input_dst], [target_src[0], target_src[0], target_src[1], target_src[1]]

    def __iter__(self):
        return self

    def __len__(self):
        return min(self.generator_src.size(), self.generator_tgt.size())
