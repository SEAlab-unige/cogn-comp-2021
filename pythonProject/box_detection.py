"""
This file contains methods to perform object detection for a single image.
It returns boxes images and/or boxes coordinates with a score greater than threshold.

Author: Tommaso Apicella
Email: tommaso.apicella@edu.unige.it

Some sections of these functions are taken from Object Detection API tutorial.
"""

import tensorflow as tf
import numpy as np
import cv2

from object_detection.utils import ops as utils_ops


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.mode == 'L':  # image is grayscale
        image_RGB = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
        return image_RGB.astype(np.uint8)
    elif image.mode == 'RGB':  # image is RGB
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    else:
        return None


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_detection_boxes(image, detection_graph):
    det_box = []
    # The array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    if image_np is not None:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # If at least one object is detected
        if output_dict['num_detections'] > 0:
            im_width, im_height = image.size
            # Take only the boxes with score greater than threshold
            for i, box in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > 0.3:
                    # Values inside box array are normalized
                    # box[0] is ymin
                    # box[1] is xmin
                    # box[2] is ymax
                    # box[3] is xmax
                    # crop takes rectangle, as a (left, upper, right, lower)-tuple
                    det_box.append(
                        image.crop((int(round(box[1] * im_width)), int(round(box[0] * im_height)),
                                    int(round(box[3] * im_width)), int(round(box[2] * im_height)))))
            return det_box
    return None


def get_detection_boxes_coord(image, detection_graph):
    det_box = []
    # The array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    if image_np is not None:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # If at least one object is detected
        if output_dict['num_detections'] > 0:
            im_width, im_height = image.size
            # Take only the boxes with score greater than threshold
            for i, box in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > 0.3:
                    # Values inside box array are normalized
                    # box[0] is ymin
                    # box[1] is xmin
                    # box[2] is ymax
                    # box[3] is xmax
                    # crop takes rectangle, as a (left, upper, right, lower)-tuple
                    det_box.append((int(round(box[1] * im_width)), int(round(box[0] * im_height)),
                                    int(round(box[3] * im_width)), int(round(box[2] * im_height))))
            return det_box
    return None


def get_detection_boxes_and_coord(image, detection_graph):
    det_box = []
    det_box_coord = []
    # The array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    if image_np is not None:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # If at least one object is detected
        if output_dict['num_detections'] > 0:
            im_width, im_height = image.size
            # Take only the boxes with score greater than threshold
            for i, box in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > 0.3:
                    # Values inside box array are normalized
                    # box[0] is ymin
                    # box[1] is xmin
                    # box[2] is ymax
                    # box[3] is xmax
                    # crop takes rectangle, as a (left, upper, right, lower)-tuple
                    det_box.append(
                        image.crop((int(round(box[1] * im_width)), int(round(box[0] * im_height)),
                                    int(round(box[3] * im_width)), int(round(box[2] * im_height)))))
                    det_box_coord.append((int(round(box[1] * im_width)), int(round(box[0] * im_height)),
                                          int(round(box[3] * im_width)), int(round(box[2] * im_height))))
            return det_box, det_box_coord
    return None


def get_detection_boxes_threshold(image, detection_graph, thresh):
    det_box = []
    # The array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    if image_np is not None:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # If at least one object is detected
        if output_dict['num_detections'] > 0:
            im_width, im_height = image.size
            # Take only the boxes with score greater than threshold
            for i, box in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > thresh:
                    # Values inside box array are normalized
                    # box[0] is ymin
                    # box[1] is xmin
                    # box[2] is ymax
                    # box[3] is xmax
                    # crop takes rectangle, as a (left, upper, right, lower)-tuple
                    det_box.append(
                        image.crop((int(round(box[1] * im_width)), int(round(box[0] * im_height)),
                                    int(round(box[3] * im_width)), int(round(box[2] * im_height)))))
            return det_box
    return None
