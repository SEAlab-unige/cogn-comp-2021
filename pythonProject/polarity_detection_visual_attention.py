"""
This script employs the algorithm described by Ragusa et al., "Design and Deployment of an Image Polarity Detector with Visual Attention", to detect the polarity of images in a directory.

Author: Tommaso Apicella
Email: tommaso.apicella@edu.unige.it
"""

import tensorflow as tf
import box_detection as od
import numpy as np
import os
import write_file as wf

from preprocessing import preprocessing_factory as pf
from nets import nets_factory
from PIL import Image

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('net_name', "mobilenet_v1_1.0_224", "The name of the net.")
flags.DEFINE_integer('image_size', 224, "Size of the placeholder.")
flags.DEFINE_string('path_to_polarity_ckpt_file',
                    None,
                    "Path to entire image polarity classificator checkpoint file.")
flags.DEFINE_string('path_to_finetuned_polarity_ckpt_file',
                    None,
                    "Path to finetuned polarity classificator checkpoint file.")
flags.DEFINE_string('train_set_name', "anp40", "Training set output classes.")
flags.DEFINE_string('path_to_res_file', None, "Path to results file.")
flags.DEFINE_string('path_to_images_dir', None,
                    "Path to directory containing images.")
flags.DEFINE_string('path_to_images_labels', None,
                    "Path to file containing ground truth labels.")
flags.DEFINE_string('path_to_obj_det_finetuned',
                    None,
                    "Path to fine-tuned object detector frozen inference graph file.")
FLAGS = flags.FLAGS

net_names = {
    'mobilenet_v1_1.0_224': 'mobilenet_v1'
}

preprocessings_map = {
    'mobilenet_v1_1.0_224': 'mobilenet_v1',
}

model_variables = {
    'mobilenet_v1_1.0_224': 'MobilenetV1',
}

labels = {
    't4sa': {0: 'negative', 1: 'neutral', 2: 'positive'},
    'mvso40': {0: 'negative', 1: 'positive'},
    'anp40': {0: 'negative', 1: 'positive'},
}


def get_filenames(dataset_dir, extension):
    # Store all filenames with extension in a list
    filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if path.endswith(extension):
            filenames.append(path)

    return filenames


def retrieve_ground_truth_labels(images_path, labels_path):
    # Store labels in a list (the order is the same of the images)
    labels_list = []
    file = open(labels_path, "r")
    content = file.readlines()
    # Images and labels could be stored in different order
    for image in images_path:
        # For each image
        image_name = os.path.basename(image)
        for line in content:
            # Look for the corresponding label
            if line.split()[0] == image_name:
                labels_list.append(int(line.split()[1]))
                break
    file.close()
    return labels_list


def modify_labels_and_images(images_path, gt_labels):
    # Retrieve indices of ground truth third class
    index_list = [i for i, el in enumerate(gt_labels) if el == 1]
    # Remove elements
    images_path_temp = [path for i, path in enumerate(images_path) if i not in index_list]
    gt_labels_temp = [label for i, label in enumerate(gt_labels) if i not in index_list]
    # Change elements
    for i, el in enumerate(gt_labels_temp):
        if el == 2:
            gt_labels_temp[i] = 1
    return images_path_temp, gt_labels_temp


def load_object_det_model(path_to_frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def fill_confidence_bin(conf_bin, pol_pred_box, ov_perc):
    # Set right sign
    conf_temp = 0.0
    if pol_pred_box[0] == 0:
        conf_temp = (-1) * pol_pred_box[1]
    else:
        conf_temp = pol_pred_box[1]
    # Select correct bin
    for i in range(0, conf_bin.shape[1]):
        if ov_perc < ((i + 1) * (1.0 / conf_bin.shape[1])):
            conf_bin[0][i] += conf_temp
            break


def retrieve_index(conf_bin):
    abs_array = np.abs(conf_bin[0][:])
    return np.argmax(abs_array)


def main(_):
    # Assertions
    assert FLAGS.path_to_polarity_ckpt_file is not None, "Please, provide the path to image polarity classifier's checkpoint file"
    assert FLAGS.path_to_finetuned_polarity_ckpt_file is not None, "Please, provide the path to finetuned polarity classifier's checkpoint file"
    assert FLAGS.path_to_obj_det_finetuned is not None, "Please, provide the path to fine-tuned object detector frozen inference graph file"
    assert os.path.exists(FLAGS.path_to_obj_det_finetuned), "Path to fine-tuned object detector frozen inference graph file does not exist"
    assert FLAGS.path_to_images_dir is not None, "Please, provide the path to images directory"
    assert os.path.exists(FLAGS.path_to_images_dir), "Path to images directory does not exist"
    assert FLAGS.path_to_images_labels is not None, "Please, provide the path to images labels file"
    assert os.path.exists(FLAGS.path_to_images_labels), "Path to images labels file does not exist"
    assert FLAGS.path_to_res_file is not None, "Please, provide the path to result file"

    # Retrieve images path
    images_path = get_filenames(FLAGS.path_to_images_dir, '.jpg')
    # Retrieve ground truth labels
    gt_labels = retrieve_ground_truth_labels(images_path, FLAGS.path_to_images_labels)
    # Understand if the dataset has three classes
    if gt_labels.__contains__(0) and gt_labels.__contains__(1) and gt_labels.__contains__(2):
        images_path, gt_labels = modify_labels_and_images(images_path, gt_labels)

    # Load categories
    categories = list(labels[FLAGS.train_set_name].values())
    preprocessing = preprocessings_map[FLAGS.net_name]
    object_det_model_finetuned = load_object_det_model(FLAGS.path_to_obj_det_finetuned)
    with tf.Graph().as_default():
        print("---------- GRAPH ----------")
        # Preprocessing node
        image = tf.placeholder(tf.uint8, shape=[None, None, 3])
        preprocessing_fn = pf.get_preprocessing(preprocessing)
        processed_image = preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
        # Adding one dimension for convention
        processed_images = tf.expand_dims(processed_image, 0)
        # Retrieve net
        network_fn = nets_factory.get_network_fn(net_names[FLAGS.net_name], len(categories))
        logits, nodes = network_fn(processed_images)
        # Get class
        output = tf.argmax(logits, 1)
        # Get probability
        probabilities = tf.nn.softmax(logits)
        model_variable = model_variables[FLAGS.net_name]
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.path_to_polarity_ckpt_file,
                                                 slim.get_model_variables(model_variable))
        init_fn_2 = slim.assign_from_checkpoint_fn(FLAGS.path_to_finetuned_polarity_ckpt_file,
                                                   slim.get_model_variables(model_variable))

        pred = []
        for index, i in enumerate(images_path):
            im_temp = Image.open(i)
            if im_temp.mode == 'L':
                im = Image.Image.convert(im_temp, 'RGB')
            else:
                im = im_temp

            ##### Entire image #####
            width_entire_im, height_entire_im = im.size
            with tf.Session() as sess:
                print("---------- SESSION ----------")
                # Load weights
                init_fn(sess)
                # Perform prediction
                entire_im_output, entire_im_prob = sess.run([output, probabilities], feed_dict={image: im})
                entire_im_pred = [entire_im_output[0], entire_im_prob[0][entire_im_output[0]]]
                # Close session
                sess.close()

            ##### Saliency detector #####
            # Retrieve detection boxes in image
            det_boxes = od.get_detection_boxes_threshold(im, object_det_model_finetuned, 0.3)
            # If there are some detection boxes
            if det_boxes is not None and len(det_boxes) > 0:
                NUM_BINS = 5
                conf_bin = np.zeros(shape=(1, NUM_BINS))
                # For each detection box
                for det_box in det_boxes:
                    # Compute overlapping percentage with respect to entire image
                    width_det_box, height_det_box = det_box.size
                    ov_percentage = (width_det_box * height_det_box) / (width_entire_im * height_entire_im)
                    # Run polarity inference
                    with tf.Session() as sess_2:
                        print("---------- SESSION ----------")
                        # Load weights
                        init_fn_2(sess_2)
                        det_box_output, det_box_prob = sess_2.run([output, probabilities], feed_dict={image: det_box})
                        det_box_pred = [det_box_output[0], det_box_prob[0][det_box_output[0]]]
                        fill_confidence_bin(conf_bin, det_box_pred, ov_percentage)
                        # Close session
                        sess_2.close()
                # Compare best bin with prediction on entire image
                best_index = retrieve_index(conf_bin)
                if np.abs(conf_bin[0][best_index]) > entire_im_pred[1]:
                    if conf_bin[0][best_index] < 0:
                        pred.append(0)
                    else:
                        pred.append(1)
                else:
                    pred.append(entire_im_pred[0])
            else:
                pred.append(entire_im_pred[0])
            print("{}/{}".format(index + 1, len(images_path)))
        # Write results in output file
        wf.write_confusion_matrix_in_file(FLAGS.path_to_res_file, gt_labels, pred)


if __name__ == '__main__':
    tf.app.run()
