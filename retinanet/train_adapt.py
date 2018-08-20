"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import functools
import os
import sys
import numpy as np
import warnings
import logging
import keras
import keras.preprocessing.image
import tensorflow as tf
import progressbar
from sklearn.preprocessing import normalize

from keras.layers import Lambda
from keras_retinanet import losses
from keras_retinanet import layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.visualization import draw_annotations
import apollo_python_common.io_utils as io_utils
from retinanet.traffic_signs_eval import TrafficSignsEval
from utils import Logger as TFLogger

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, backbone, num_classes, weights, tensorboard, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    src_inputs = keras.layers.Input(shape=(None, None, 3), name='src_inputs')
    dst_inputs = keras.layers.Input(shape=(None, None, 3), name='dst_inputs')
    model_inference, model_G, model_C1, model_C2 = backbone_retinanet(
        num_classes, backbone=backbone, nms=True, modifier=modifier, adapt=True)
    model_G = model_with_weights(model_G, weights=weights, skip_mismatch=True)
    logging.debug('\n-----------\n| model_G |\n-----------')
    logging.debug(model_G.summary(print_fn=logging.debug))
    logging.debug('\n------------\n| model_C1 |\n------------')
    logging.debug(model_C1.summary(print_fn=logging.debug))
    logging.debug('\n------------\n| model_C2 |\n------------')
    logging.debug(model_C2.summary(print_fn=logging.debug))

    src_G_outputs = model_G(src_inputs)
    src_regr = Lambda(lambda x: x, name='src_regression')(src_G_outputs[0])
    src_C1_clas = Lambda(lambda x: x, name='src_C1_classification')(model_C1(src_G_outputs[1:]))
    src_C2_clas = Lambda(lambda x: x, name='src_C2_classification')(model_C2(src_G_outputs[1:]))

    dst_G_outputs = model_G(dst_inputs)
    dst_regr = Lambda(lambda x: x, name='dst_regression')(dst_G_outputs[0])
    dst_C1_clas = Lambda(lambda x: x, name='dst_C1_classification')(model_C1(dst_G_outputs[1:]))
    dst_C2_clas = Lambda(lambda x: x, name='dst_C2_classification')(model_C2(dst_G_outputs[1:]))

    src_discr_clas = losses.DiscrepancyClas(name='src_discr_clas')([src_C1_clas, src_C2_clas])
    src_neg_discr_clas = Lambda(lambda x: -x, name='src_neg_discr_clas')(src_discr_clas)

    dst_discr_clas = losses.DiscrepancyClas(name='dst_discr_clas')([dst_C1_clas, dst_C2_clas])
    dst_neg_discr_clas = Lambda(lambda x: -x, name='dst_neg_discr_clas')(dst_discr_clas)

    # Step 1.
    inputs = [src_inputs]
    outputs = [src_regr, src_C1_clas, src_C2_clas]
    model_step1 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step1')
    model_step1.compile(
        loss={
            'src_regression'       : losses.smooth_l1(),
            'src_C1_classification': losses.focal(),
            'src_C2_classification': losses.focal(),
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    logging.debug('\n---------------\n| model_step1 |\n---------------')
    logging.debug(model_step1.summary(print_fn=logging.debug))

    # Step 2.
    # How to use make a part of the model non-trainable:
    #   https://gist.github.com/naotokui/a9274f12af9d946e99b6df73a5d2af6d
    model_G.trainable = False
    model_C1.trainable = True
    model_C2.trainable = True
    inputs = [src_inputs, dst_inputs]
    outputs = [src_regr, src_C1_clas, src_C2_clas, dst_neg_discr_clas]
    model_step2 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step2')
    model_step2.compile(
        loss={
            'src_regression'       : losses.smooth_l1(),
            'src_C1_classification': losses.focal(),
            'src_C2_classification': losses.focal(),
            'dst_neg_discr_clas'   : losses.zero_loss,
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    logging.debug('\n---------------\n| model_step2 |\n---------------')
    logging.debug(model_step2.summary(print_fn=logging.debug))

    # Step 3.
    model_G.trainable = True
    model_G.get_layer('regression_submodel').trainable = False
    model_C1.trainable = False
    model_C2.trainable = False
    inputs = [src_inputs, dst_inputs]
    outputs = [src_regr, src_C1_clas, src_C2_clas, dst_discr_clas]
    model_step3 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step3')
    model_step3.compile(
        loss={
            'dst_discr_clas'       : losses.zero_loss,
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    logging.debug('\n---------------\n| model_step3 |\n---------------')
    logging.debug('model_step3')
    logging.debug(model_step3.summary(print_fn=logging.debug))

    # Predict.
    inputs = [dst_inputs]
    outputs = [dst_regr, dst_C1_clas, dst_C2_clas, dst_discr_clas]
    model_inference = keras.models.Model(inputs=inputs, outputs=outputs, name='inference')
    
    return model_step1, model_step2, model_step3, model_inference, model_G


def create_generators(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(min_rotation=-0.122173,
                                                     max_rotation=0.122173)

    if args.dataset_type == 'traffic_signs':
        from retinanet.traffic_signs_generator import TrafficSignsGenerator
        train_src_generator = TrafficSignsGenerator(
            args.train_src_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            group_method='random',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    elif args.dataset_type == 'csv':
        from keras_retinanet.preprocessing.csv_generator import CSVGenerator
        train_src_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            group_method='random'
        )

    # Currently dst data is only in ImageFolder format. Maybe make options for the future.
    from retinanet.adapt_generator import ImageFolderGenerator
    train_dst_generator = ImageFolderGenerator(
        args.train_dst_path,
        transform_generator=transform_generator,
        batch_size=args.batch_size,
        group_method='random',
        image_min_side=1080,
        image_max_side=2592
    )

    # Combine generators.
    from retinanet.adapt_generator import AdaptGenerator
    train_generator = AdaptGenerator(train_src_generator, train_dst_generator)

    return train_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.snapshot_dir and not os.path.exists(parsed_args.snapshot_dir):
        os.makedirs(parsed_args.snapshot_dir)
    elif parsed_args.snapshot_dir is None:
        logging.warning('Snapshots will not be made during training.')

    for step in parsed_args.model_steps:
        assert step in [1, 2, 3], 'Bad model_step %d' % step
    assert len(parsed_args.model_steps) >= 1, 'Model steps is empty.'

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    if 'resnet' in parsed_args.backbone:
        from keras_retinanet.models.resnet import validate_backbone
    elif 'mobilenet' in parsed_args.backbone:
        from keras_retinanet.models.mobilenet import validate_backbone
    elif 'vgg' in parsed_args.backbone:
        from keras_retinanet.models.vgg import validate_backbone
    elif 'densenet' in parsed_args.backbone:
        from keras_retinanet.models.densenet import validate_backbone
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(parsed_args.backbone))

    validate_backbone(parsed_args.backbone)

    return parsed_args


def parse_args():
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    ts_parser = subparsers.add_parser('traffic_signs')
    ts_parser.add_argument('train_src_path', help='Path to folder containing files used for train. The rois.bin file should be there.')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    parser.add_argument('train_dst_path', help='Path to folder with dst domain images.')

    parser.add_argument('--load-snapshot',   help='Resume training from a snapshot. If not specified, will use ImageNet weights.')
    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=1080)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=2592)
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--batches-per-epoch', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-dir',    help='Dir to store snapshots of models, specify "" for no snapshpts.', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output, if not specified, no logging.')
    parser.add_argument('--tensorboard-freq', type=int, default=10, help='how often to log changes to tensorboard.')
    parser.add_argument('--tensorboard-offset', help='The offset for reporting in tensorboard.', type=int, default=0)
    parser.add_argument('--logging',         default=20, type=int, choices=[10, 20, 30, 40], help='Log debug (10), info (20), warning (30), error (40).')
    parser.add_argument('--model-steps',     type=lambda s: [int(item) for item in s.split(',')], default='1,2,3', help='Debugging only: which out of the three steps to run.')

    return check_args(parser.parse_args())


def log_G_weights_distrib(tensorboard, model, ibatch):
  head_name = 'G'
  for layer in model.get_layer(head_name).layers:
    if hasattr(layer, 'layers'):
      for layerin in layer.layers:
        assert not hasattr(layerin, 'layers'), layerin.layers
        if layerin.get_weights():
#          print ('%s/%s/%s' % (head_name, layer.name, layerin.name))
          tensorboard.log_histogram('%s/%s/%s/weights' % (head_name, layer.name, layerin.name), layerin.get_weights()[0], ibatch)
          tensorboard.log_histogram('%s/%s/%s/biases' % (head_name, layer.name, layerin.name), layerin.get_weights()[1], ibatch)

def log_head_weights_distrib(tensorboard, model, head_name, ibatch):
  for layer in model.get_layer(head_name).layers:
    if hasattr(layer, 'layers'):
      for layerin in layer.layers:
        assert not hasattr(layerin, 'layers'), layerin.layers
        if layerin.get_weights():
#          print ('%s/%s/%s' % (head_name, layer.name, layerin.name))
          tensorboard.log_histogram('%s/%s/%s/weights' % (head_name, layer.name, layerin.name), layerin.get_weights()[0], ibatch)
#          tensorboard.log_histogram('%s/%s/%s/biases' % (head_name, layer.name, layerin.name), layerin.get_weights()[1], ibatch)
#          biases_as_grid = normalize(layerin.get_weights()[1].copy().reshape((9,10)), norm='l1')[np.newaxis,:,:] / 3. + 0.5
#          tensorboard.log_images('model/%s/%s/biases' % (head_name, layerin.name), biases_as_grid, ibatch)


def main():
    progressbar.streams.wrap_stderr()
    args = parse_args()
    logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # make sure keras is the minimum required version
    check_keras_version()

    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator = create_generators(args)

    if 'resnet' in args.backbone:
        from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'mobilenet' in args.backbone:
        from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'vgg' in args.backbone:
        from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
    elif 'densenet' in args.backbone:
        from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(args.backbone))

    tensorboard = TFLogger(args.tensorboard_dir)

    # Create the model.
    logger.info('Creating model, this may take a second...')
    model_step1, model_step2, model_step3, model_inference, model_G = create_models(
        backbone_retinanet=retinanet,
        backbone=args.backbone,
        num_classes=train_generator.generator_src.num_classes(),
        weights=(download_imagenet(args.backbone) if not args.load_snapshot else None),
        freeze_backbone=args.freeze_backbone,
        tensorboard=tensorboard,
    )
    if args.load_snapshot:
        logger.info('Loading model, this may take a second...')
        model_inference.load_weights(args.load_snapshot, by_name=True)

    for epoch in range(args.epochs):
        for ibatch, (inputs, targets, labels) in progressbar.ProgressBar()(enumerate(train_generator)):
            if ibatch * args.batch_size > args.batches_per_epoch:
                break
            iglobal = epoch * args.batches_per_epoch + ibatch + args.tensorboard_offset

            if ibatch % (args.tensorboard_freq * 10) == 0:  # Log images 10 less often than everything else.

                # Draw annotations on the source image and display both the source and the target images.
                src_images_with_boxes = []
                for image, annotations in zip(inputs['src'], labels['src']):
                    src_image_with_boxes = ((image.copy() + 1) * 127.5).astype(np.uint8)[:,:,::-1].copy()
                    draw_annotations(src_image_with_boxes, annotations, (0,255,0), train_generator.generator_src)
                    src_images_with_boxes.append(src_image_with_boxes)
                tensorboard.log_images('inputs/src', src_images_with_boxes, iglobal)
                tensorboard.log_images('inputs/dst', (inputs['dst'][:,:,:,::-1] / 2.0 + 0.5), iglobal)

                # Draw t-SNE
  

            # Distributions of all weights in heads C1 and in C2.
            if ibatch % args.tensorboard_freq == 0:
                log_G_weights_distrib(tensorboard, model_step1, iglobal)
                log_head_weights_distrib(tensorboard, model_step1, 'C1', iglobal)
                log_head_weights_distrib(tensorboard, model_step1, 'C2', iglobal)

                predict = model_inference.predict(x=[inputs['src']])
                tensorboard.log_histogram('predict/src_C1', predict[1].flatten(), iglobal)
                tensorboard.log_histogram('predict/src_C2', predict[2].flatten(), iglobal)
                tensorboard.log_histogram('predict/src_discr', predict[3].flatten(), iglobal)
                predict = model_inference.predict(x=[inputs['dst']])
                tensorboard.log_histogram('predict/dst_C1', predict[1].flatten(), iglobal)
                tensorboard.log_histogram('predict/dst_C2', predict[2].flatten(), iglobal)
                tensorboard.log_histogram('predict/dst_discr', predict[3].flatten(), iglobal)

            if 1 in args.model_steps:
                losses1 = model_step1.train_on_batch(
                    x=[inputs['src']],
                    y=[targets['src'][0], targets['src'][1], targets['src'][1]])
                if ibatch % args.tensorboard_freq == 0:
                    for loss_name, loss in zip(model_step1.metrics_names, losses1):
                        tensorboard.log_scalar('step1/' + loss_name, loss, iglobal)

            if 2 in args.model_steps:
                losses2 = model_step2.train_on_batch(
                    x=[inputs['src'], inputs['dst']],
                    y=[targets['src'][0], targets['src'][1], targets['src'][1], np.zeros((1,))])
                if ibatch % args.tensorboard_freq == 0:
                    for loss_name, loss in zip(model_step2.metrics_names, losses2):
                        tensorboard.log_scalar('step2/' + loss_name, loss, iglobal)

            if 3 in args.model_steps:
                losses3 = model_step3.train_on_batch(
                    x=[inputs['src'], inputs['dst']],
                    y=[np.zeros((1,))])
                if ibatch % args.tensorboard_freq == 0:
                    for loss_name, loss in zip(model_step3.metrics_names, losses3):
                        tensorboard.log_scalar('step3/' + loss_name, loss, iglobal)

        if args.snapshot_dir:
            model_inference.save(os.path.join(args.snapshot_dir, 'epoch%03d-step%d.h5' % (epoch, iglobal)))

if __name__ == '__main__':
    main()
