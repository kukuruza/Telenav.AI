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
from keras.utils import multi_gpu_model
import tensorflow as tf
import progressbar

from keras.layers import Lambda
from keras_retinanet import losses
from keras_retinanet import layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet.utils.model import freeze as freeze_model
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from retinanet.traffic_signs_generator import TrafficSignsGenerator
from retinanet.traffic_signs_eval import TrafficSignsEval
from retinanet.adapt_generator import ImageFolderGenerator, AdaptGenerator
from utils import Logger

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, backbone, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    src_inputs = keras.layers.Input(shape=(None, None, 3), name='src_inputs')
    dst_inputs = keras.layers.Input(shape=(None, None, 3), name='dst_inputs')
    if multi_gpu > 1:
        assert False
    else:
        retina_model, model_G, model_C = backbone_retinanet(
            num_classes, backbone=backbone, nms=True, modifier=modifier, adapt=True)
        model_C = model_with_weights(model_C, weights=weights, skip_mismatch=True)
        logging.info(model_C.summary())

    src_G_outputs = model_G(src_inputs)
    src_C_outputs = model_C(src_G_outputs)
    src_C1_regr = Lambda(lambda x: x, name='src_C1_regression')(src_C_outputs[0])
    src_C2_regr = Lambda(lambda x: x, name='src_C2_regression')(src_C_outputs[1])
    src_C1_clas = Lambda(lambda x: x, name='src_C1_classification')(src_C_outputs[2])
    src_C2_clas = Lambda(lambda x: x, name='src_C2_classification')(src_C_outputs[3])

    dst_G_outputs = model_G(dst_inputs)
    dst_C_outputs = model_C(dst_G_outputs)
    dst_C1_regr = Lambda(lambda x: x, name='dst_C1_regression')(dst_C_outputs[0])
    dst_C2_regr = Lambda(lambda x: x, name='dst_C2_regression')(dst_C_outputs[1])
    dst_C1_clas = Lambda(lambda x: x, name='dst_C1_classification')(dst_C_outputs[2])
    dst_C2_clas = Lambda(lambda x: x, name='dst_C2_classification')(dst_C_outputs[3])

    # Step 1.
    inputs = [src_inputs]
    outputs = [src_C1_regr, src_C2_regr, src_C1_clas, src_C2_clas]
    model_step1 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step1')
    model_step1.compile(
        loss={
            'src_C1_regression'    : losses.smooth_l1(),
            'src_C2_regression'    : losses.smooth_l1(),
            'src_C1_classification': losses.focal(),
            'src_C2_classification': losses.focal(),
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    logging.info('model_step1')
    logging.info(model_step1.summary())

    # Step 2.
    # How to use make a part of the model non-trainable:
    #   https://gist.github.com/naotokui/a9274f12af9d946e99b6df73a5d2af6d
    inputs = [src_inputs, dst_inputs]
    #dst_neg_discr_clas = losses.FocalDiscrepancyClas(name='dst_neg_discr_clas', gamma=2, negative=True)([dst_C1_clas, dst_C2_clas])
    dst_neg_discr_clas = losses.DiscrepancyClas(name='dst_neg_discr_clas', negative=True, alpha=1)([dst_C1_clas, dst_C2_clas])
    print ('dst_neg_discr_clas', dst_neg_discr_clas.get_shape())
    outputs = [src_C1_regr, src_C2_regr, src_C1_clas, src_C2_clas, dst_neg_discr_clas]
    model_G.trainable = False
    model_C.trainable = True
    model_step2 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step2')
    model_step2.compile(
        loss={
            'src_C1_regression'    : losses.smooth_l1(),
            'src_C2_regression'    : losses.smooth_l1(),
            'src_C1_classification': losses.focal(),
            'src_C2_classification': losses.focal(),
            'dst_neg_discr_clas'   : losses.zero_loss,
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    #logging.info('model_step2')
    #logging.info(model_step2.summary())

    # Step 3.
    inputs = [dst_inputs]
    #dst_discr_clas = losses.FocalDiscrepancyClas(name='dst_discr_clas', gamma=2)([dst_C1_clas, dst_C2_clas])
    dst_discr_clas = losses.DiscrepancyClas(name='dst_discr_clas', alpha=1)([dst_C1_clas, dst_C2_clas])
    print ('dst_discr_clas', dst_discr_clas.get_shape())
    outputs = [dst_discr_clas, dst_C1_clas, dst_C2_clas]
    model_G.trainable = True
    model_C.trainable = False
    model_step3 = keras.models.Model(inputs=inputs, outputs=outputs, name='adapt-step3')
    model_step3.compile(
        loss={
            'dst_discr_clas'       : losses.zero_loss,
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    logging.info('model_step3')
    logging.info(model_step3.summary())

    return retina_model, model_step1, model_step2, model_step3


def create_callbacks(args):
    callbacks = {}

    # save the prediction model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        io_utils.create_folder(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        #checkpoint = RedirectModel(checkpoint, prediction_model)
        callbacks['snapshots'] = checkpoint

    callbacks['ReduceLROnPlateau'] = keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    )

    return callbacks


def create_generators(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(min_rotation=-0.122173,
                                                     max_rotation=0.122173)
    train_generator = None

    if args.dataset_type == 'traffic_signs':
        train_src_generator = TrafficSignsGenerator(
            args.train_src_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            group_method='random',
            image_min_side=1080,
            image_max_side=2592
        )
        train_dst_generator = ImageFolderGenerator(
            args.train_dst_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            group_method='random',
            image_min_side=1080,
            image_max_side=2592
        )
        # Combine generators.
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

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

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


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    ts_parser = subparsers.add_parser('traffic_signs')
    ts_parser.add_argument('train_src_path', help='Path to folder containing files used for train. The rois.bin file should be there.')
    ts_parser.add_argument('train_dst_path', help='Path to folder containing files used for train. The rois.bin file should be there.')

    def csv_list(string):
        return string.split(',')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=1)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')

    return check_args(parser.parse_args(args))


def main(args=None):
    progressbar.streams.wrap_stderr()
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator = create_generators(args)

    if 'resnet' in args.backbone:
        from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'mobilenet' in args.backbone:
#        from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'vgg' in args.backbone:
#        from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'densenet' in args.backbone:
#        from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(args.backbone))

    # create the model
    if args.snapshot is not None:
        logger.info('Loading model, this may take a second...')
        model            = keras.models.load_model(args.snapshot, custom_objects=custom_objects)
        prediction_model = model
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = download_imagenet(args.backbone)

        logger.info('Creating model, this may take a second...')
        model, model_step1, model_step2, model_step3 = create_models(
            backbone_retinanet=retinanet,
            backbone=args.backbone,
            num_classes=train_generator.generator_src.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone
        )

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets

    # start training
#    training_model.fit_generator(
#        generator=train_generator,
#        steps_per_epoch=args.steps,
#        epochs=args.epochs,
#        verbose=1,
#        callbacks=callbacks,
#        workers=4
#    )

    losses_names1 = model_step1.metrics_names
    losses_names2 = model_step2.metrics_names
    losses_names3 = model_step3.metrics_names
    logger.info('losses in steps \n\t1: %s \n\t2: %s \n\t3: %s' % 
        (losses_names1, losses_names2, losses_names3))

    zeros = np.zeros((args.batch_size,1,1))

    # https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    tensorboard = Logger(args.tensorboard_dir)
 
    for epoch in range(args.epochs):
        for ibatch, (inputs, targets) in progressbar.ProgressBar()(enumerate(train_generator)):

            losses1 = model_step1.train_on_batch(
                x=[inputs['src']],
                y=[targets['src'][0], targets['src'][0], targets['src'][1], targets['src'][1]])
            logger.info('1: %s' % [str(x) for x in zip(losses_names1, losses1)])
            for loss_name, loss in zip(losses_names1, losses1):
                tensorboard.log_scalar('step1/' + loss_name, loss, ibatch)

#            losses2 = model_step2.train_on_batch(
#                x=[inputs['src'], inputs['dst']],
#                y=[targets['src'][0], targets['src'][0], targets['src'][1], targets['src'][1], zeros])
#            logger.info('2: %s' % [str(x) for x in zip(losses_names2, losses2)])
#            for loss_name, loss in zip(losses_names2, losses2):
#                tensorboard.log_scalar('step2/' + loss_name, loss, ibatch)
#            predict2 = model_step2.predict(x=[inputs['src'], inputs['dst']])
#            tensorboard.log_histogram('step2/dst_neg_discr', predict2[4].flatten(), ibatch)

#            losses3 = model_step3.train_on_batch(
#                x=[inputs['dst']],
#                y=[zeros])
#            logger.info('3: %s' % str((losses_names3, losses3)))
#            for loss_name, loss in zip(losses_names3, losses3):
#                tensorboard.log_scalar('step3/' + loss_name, loss, ibatch)
#            predict3 = model_step3.predict(x=[inputs['dst']])
#            tensorboard.log_histogram('step3/dst_discr', predict3[0].flatten(), ibatch)
#            tensorboard.log_histogram('step3/dst_C1', predict3[1].flatten(), ibatch)
#            tensorboard.log_histogram('step3/dst_C2', predict3[2].flatten(), ibatch)


if __name__ == '__main__':
    main()
