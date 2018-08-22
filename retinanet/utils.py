"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os
import sys
import logging
import cv2
import numpy as np
import time
import itertools
import traceback
import math
import threading
import keras_retinanet.utils.image as retinanet_image
import keras.backend as K
from collections import OrderedDict
from keras.models import Model as Keras_Model
from tensorflow import Session as Tensorflow_Session
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from io import BytesIO
import matplotlib.pyplot as plt

import apollo_python_common.image
from vanishing_point.vanishing_point import VanishingPointDetector
import apollo_python_common.io_utils as io_utils
from apollo_python_common.rectangle import Rectangle
import apollo_python_common.proto_api as meta

RESOLUTIONS_COLORS = [(0, 0, 255), (255, 0, 0), (255,255,0), (0,255,255), (255,0,255)]
VP_DETECTOR = VanishingPointDetector()
VP_CONFIDENCE_THRESHOLD = 0.2
VP_SIGNIFICATIVE_Y_PERCENTAGE = 1.15
MIN_SCORE_THRESHOLD = 0.05
MAX_DETECTIONS=300


class Logger(object):
    """
    Logging in tensorboard without tensorflow ops.
    https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.log_dir = log_dir
        if log_dir is None:
            logging.warning('Tensorboard is disabled.')
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        if self.writer is None:
            return

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        if self.writer is None:
            return

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        
    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        if self.writer is None:
            return

        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_projection(self, tag, values, labels, step):
        if self.writer is None:
            return

        sess = K.get_session()
        with tf.device("/cpu:0"):
            values = tf.Variable(values, name=tag)
        sess.run(values.initializer)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = tag + ':0'
        metadata_name = 'projection-step%d.tsv' % step
        embed.metadata_path = metadata_name
        #embed.sprite.image_path = os.path.join(FLAGS.data_dir + '/mnist_10k_sprite.png')
        projector.visualize_embeddings(self.writer, config)

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.log_dir, 'projection-step%d' % step), 1)

        if labels is not None:
            with open(os.path.join(self.log_dir, metadata_name), 'w') as f:
                f.write('\n'.join([str(x) for x in labels.flatten().tolist()]))


def predict_folder(model, input_folder, output_folder, resolutions, rois_labels, score_threshold_per_class,
                   algorithm="retinanet",draw_predictions=False, max_number_of_images=None, log_level=0):
    logger = logging.getLogger(__name__)
    if log_level > 0:
        logger.info('Prediction on directory {} is starting.'.format(input_folder))
    if output_folder is not None and not os.path.isdir(output_folder):
        io_utils.create_folder(output_folder)

    img_files_lst = io_utils.get_images_from_folder(input_folder)
    if max_number_of_images!=None:
        img_files_lst = img_files_lst[:max_number_of_images]
    all_predictions = dict()
    for idx, file_name in enumerate(img_files_lst):
        if log_level > 0:
            logger.info('{}/{} {}'.format(idx, len(img_files_lst), file_name))
        try:
            image = apollo_python_common.image.get_bgr(file_name)
            if image is not None:
                image, image_pred = preprocess_image(image)
                merged_predictions, predictions_per_resolutions = predict_one_image(image_pred, model, rois_labels,
                                                                                    resolutions, score_threshold_per_class,
                                                                                    log_level)
                all_predictions[os.path.basename(file_name)] = merged_predictions
                if draw_predictions:
                    # Painting predictions
                    paint_detections_to_image(image, merged_predictions, (0, 0, 255))
                    # # For debugging:
                    # for idx, prediction_per_resolution in enumerate(predictions_per_resolutions):
                    #     paint_detections_to_image(image, prediction_per_resolution, RESOLUTIONS_COLORS[idx])
                    out_file_name = os.path.join(output_folder, os.path.basename(file_name))
                    # Saving image with predicted boxes
                    cv2.imwrite(out_file_name, image)
            else:
                logger.warning('Image {} is corrupted'.format(file_name))
        except Exception as err:
            logger.error(err)
            print(traceback.format_exc())

    metadata = get_preds_in_common_format(all_predictions, algorithm, "")
    return metadata


def preprocess_image(image):
    image_cropped = get_image_fc_VP(image)
    image_pred = retinanet_image.preprocess_image(image_cropped)
    return image_cropped, image_pred


def get_image_fc_VP(image):
    '''
    Gets the area of interest where traffic signs are located in a image
    :param image: full image
    :return: cropped image with the area above vanishing point
    '''
    detected_vp, confidence = VP_DETECTOR.get_vanishing_point(image)
    if confidence > VP_CONFIDENCE_THRESHOLD:
        crop_y = math.floor(detected_vp.y * VP_SIGNIFICATIVE_Y_PERCENTAGE)
        new_image = image[:crop_y, :]
        return new_image
    else:
        return image


def get_graph_name():
    return "retinanet" + str(threading.get_ident())


def get_predicted_detections(model, images_for_pred):
    if type(model) is Keras_Model:
        _, _, boxes, nms_classification = model.predict(images_for_pred)

    elif type(model) is Tensorflow_Session:
        session = model
        graph_name = get_graph_name()
        input_tensor = session.graph.get_tensor_by_name("{}/input_1:0".format(graph_name))
        regression = session.graph.get_tensor_by_name("{}/output0:0".format(graph_name))
        classification = session.graph.get_tensor_by_name("{}/output1:0".format(graph_name))
        clipped_boxes = session.graph.get_tensor_by_name("{}/output2:0".format(graph_name))
        nms = session.graph.get_tensor_by_name("{}/output3:0".format(graph_name))
        boxes, nms_classification = session.run([clipped_boxes, nms], feed_dict={input_tensor: images_for_pred})
    else:
        raise Exception('model type {} not supported'.format(str(type(model))))

    all_boxes, all_scores, all_labels = list(), list(), list()

    for idx_img in range(len(images_for_pred)):
        # select indices which have a score above the threshold
        indices = np.where(nms_classification[idx_img, :, :] > MIN_SCORE_THRESHOLD)
        # select those scores
        scores = nms_classification[idx_img][indices]
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:MAX_DETECTIONS]
        # select detections
        image_boxes = boxes[idx_img, indices[0][scores_sort], :]
        image_scores = nms_classification[idx_img, indices[0][scores_sort], indices[1][scores_sort]]
        image_predicted_labels = indices[1][scores_sort]
        all_boxes.append(image_boxes)
        all_scores.append(image_scores)
        all_labels.append(image_predicted_labels)
    return all_boxes, all_scores, all_labels


def predict_images_on_batch(images, model, rois_labels, resolutions, score_threshold_per_class, log_level):
    all_boxes, all_resolutions, all_predicted_labels, all_scores, all_predicted_label_names = \
        OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    final_predictions = list()
    for resolution in resolutions:
        scales = list()
        images_for_pred = list()
        for image_pred in images:
            image_resized, scale = get_regular_sized_image(image_pred, max_image_shape=(resolution, resolution, 3))
            scales.append(scale)
            images_for_pred.append(np.expand_dims(image_resized, axis=0))
        images_for_pred = np.vstack(images_for_pred)
        all_images_boxes, all_images_scores, all_images_labels = get_predicted_detections(model, images_for_pred)
        for i in range(len(images)):
            boxes = all_images_boxes[i] / scales[i]
            resolution = np.full(len(boxes), resolution)
            predicted_labels = all_images_labels[i]
            scores = all_images_scores[i]
            predicted_label_names = [rois_labels.label_to_name(label) for label in predicted_labels]
            all_boxes.setdefault(i, []).append(boxes)
            all_resolutions.setdefault(i, []).append(resolution)
            all_predicted_labels.setdefault(i, []).append(predicted_labels)
            all_scores.setdefault(i, []).append(scores)
            all_predicted_label_names.setdefault(i, []).append(predicted_label_names)

    for i in range(len(images)):
        all_boxes_ar = np.concatenate(all_boxes[i])
        all_resolutions_ar = np.hstack(all_resolutions[i])
        all_predicted_labels_ar = np.concatenate(all_predicted_labels[i])
        all_scores_ar = np.concatenate(all_scores[i])
        all_predicted_label_names_ar = np.concatenate(all_predicted_label_names[i])
        # Non_max_suppression
        selected_indices = non_max_suppression(all_boxes_ar, all_scores_ar, all_resolutions_ar,
                                               all_predicted_label_names_ar, score_threshold_per_class)
        selected_boxes = all_boxes_ar[selected_indices]
        selected_predicted_labels = all_predicted_labels_ar[selected_indices]
        selected_scores = all_scores_ar[selected_indices]
        selected_predicted_label_names = all_predicted_label_names_ar[selected_indices]
        final_predictions.append((selected_boxes, selected_scores, selected_predicted_label_names))
    return final_predictions


def predict_one_image(image_pred, model, rois_labels, resolutions, score_threshold_per_class, log_level):
    logger = logging.getLogger(__name__)
    start = time.time()
    all_boxes, all_resolutions, all_predicted_labels, all_scores, all_predicted_label_names = [], [], [], [], []

    for resolution in resolutions:
        image_resized, scale = get_regular_sized_image(image_pred, max_image_shape=(resolution, resolution, 3))
        all_imgs_boxes, all_imgs_scores, all_imgs_labels = get_predicted_detections(model, np.expand_dims(image_resized, axis=0))
        boxes, scores, predicted_labels = all_imgs_boxes[0], all_imgs_scores[0], all_imgs_labels[0]
        boxes = boxes / scale
        resolution = np.full(len(boxes), resolution)
        predicted_label_names = [rois_labels.label_to_name(label) for label in predicted_labels]
        all_boxes.append(boxes)
        all_resolutions.append(resolution)
        all_predicted_labels.append(predicted_labels)
        all_scores.append(scores)
        all_predicted_label_names.append(predicted_label_names)
    if log_level > 0:
        logger.info("processing time: {}".format(time.time() - start))

    all_boxes_ar = np.concatenate(all_boxes)
    all_resolutions_ar = np.hstack(all_resolutions)
    all_predicted_labels_ar = np.concatenate(all_predicted_labels)
    all_scores_ar = np.concatenate(all_scores)
    all_predicted_label_names_ar = np.concatenate(all_predicted_label_names)
    # Non_max_suppression
    selected_indices = non_max_suppression(all_boxes_ar, all_scores_ar, all_resolutions_ar,
                                           all_predicted_label_names_ar, score_threshold_per_class)
    selected_boxes = all_boxes_ar[selected_indices]
    selected_predicted_labels = all_predicted_labels_ar[selected_indices]
    selected_scores = all_scores_ar[selected_indices]
    selected_predicted_label_names = all_predicted_label_names_ar[selected_indices]

    return (selected_boxes, selected_scores, selected_predicted_label_names), \
           zip(all_boxes, all_scores, all_predicted_label_names)


def get_preds_in_common_format(all_predictions, algorithm, algorithm_version):
    metadata = meta.get_new_metadata_file()
    metadata.name = "RetinaNet"
    for file_name, (boxes, scores, label_names) in all_predictions.items():
        boxes = np.clip(boxes, 1, sys.maxsize, out=boxes)
        image_proto = metadata.images.add()
        image_proto.metadata.image_path = file_name
        image_proto.metadata.region = ""
        image_proto.metadata.trip_id = ""
        image_proto.metadata.image_index = 0
        add_detections_to_img_proto(algorithm, algorithm_version, image_proto, boxes, scores, label_names, 0)
    return metadata


def add_detections_to_img_proto(algorithm, algorithm_version, image_proto, boxes, scores, label_names, min_side_size):
    for box, score, label_name in zip(boxes, scores, label_names):
        img_height = image_proto.sensor_data.img_res.height if image_proto.sensor_data.img_res.height > 0 else sys.maxsize
        img_width = image_proto.sensor_data.img_res.width if image_proto.sensor_data.img_res.width > 0 else sys.maxsize
        tl_row = int(box[1]) if box[1] > 0 else 0
        tl_col = int(box[0]) if box[0] > 0 else 0
        br_row = int(box[3]) if box[3] < img_height else img_height
        br_col = int(box[2]) if box[2] < img_width else img_width
        if (br_col - tl_col) >= min_side_size and (br_row - tl_row) >= min_side_size:
            add_roi_to_img_proto(image_proto, algorithm, algorithm_version, label_name, score,
                                 tl_row, tl_col, br_col, br_row)


def add_roi_to_img_proto(image_proto, algorithm, algorithm_version, label_name, score, tl_row, tl_col, br_col, br_row):
    roi = image_proto.rois.add()
    roi.type = meta.get_roi_type_value(label_name)
    roi.rect.tl.row = tl_row
    roi.rect.tl.col = tl_col
    roi.rect.br.row = br_row
    roi.rect.br.col = br_col
    roi.manual = False
    roi.algorithm = algorithm
    roi.algorithm_version = algorithm_version
    detection = roi.detections.add()
    detection.type = roi.type
    detection.confidence = score


def non_max_suppression(boxes, scores, resolutions, predicted_label_names, score_threshold_per_class,
                        iou_threshold=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = set([i for i in range(len(boxes)) if scores[i] > score_threshold_per_class[predicted_label_names[i]]])
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    should_continue = True
    while should_continue:
        should_continue = False
        # NMS inter-models (inter-resolutions)
        for (i1, i2) in itertools.combinations(list(range(0, len(boxes))), 2):
            if i1 in pick and i2 in pick:
                r1 = Rectangle(x1[i1], y1[i1], x2[i1], y2[i2])
                r2 = Rectangle(x1[i2], y1[i2], x2[i2], y2[i2])
                score1 = scores[i1]
                score2 = scores[i2]
                rect_intersect = r1.get_overlapped_rect(r2)
                if rect_intersect.area() / r1.area() > iou_threshold or rect_intersect.area() / r2.area() > iou_threshold:
                    exclude_idx = i1 if score1 < score2 else i2
                    pick.remove(exclude_idx)
                    should_continue = True
    selected_indices = list(pick)
    # returning only the bounding boxes indexes that were picked
    return selected_indices


def paint_detections_to_image(image, predictions, color):
    boxes, scores, predicted_label_names = predictions
    for idx, (label, score) in enumerate(zip(predicted_label_names, scores)):
        b = boxes[idx].astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 3)
        caption = "{} {:.3f}".format(label, score)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


def get_regular_sized_image(img, max_image_shape):
    size_factor = float(max([max_image_shape[0], max_image_shape[1]])) / \
                  max([img.shape[0], img.shape[1]])
    img = cv2.resize(img, None, fx=size_factor, fy=size_factor)
    return img, size_factor
