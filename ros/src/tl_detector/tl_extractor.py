#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# Set inference graph files.
SSD_GRAPH_FILE_SITE = 'model/site_ssd_inception_v2_coco_2018_01_28_39747_v1_4/frozen_inference_graph.pb'
SSD_GRAPH_FILE_SIMULATOR = 'model/sim_ssd_inception_v2_coco_2018_01_28_39747_v1_4/frozen_inference_graph.pb'

class TLExtractor(object):
    def __init__(self, is_site):
        # Load inference graph.
        if is_site:
            self.detection_graph = self.load_graph(SSD_GRAPH_FILE_SITE)
        else:
            self.detection_graph = self.load_graph(SSD_GRAPH_FILE_SIMULATOR)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return graph
    
    def extract_traffic_light_image(self, image, confidence_cutoff=0.8):
        """Extract traffic light from camera image."""
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        return (classes, scores)

if __name__ == '__main__':

    from PIL import Image

    image = Image.open('./model/rosbag_sample.jpg')

    classes, scores = TLExtractor().extract_traffic_light_image(image)

    print classes
    print scores
