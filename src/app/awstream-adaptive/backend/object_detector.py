import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from time import sleep, perf_counter as perf_counter_s
from dds_utils import writeResult, first_phase_sub, second_phase_sub
from multiprocessing.pool import Pool

class Detector:
    classes = {
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }
    rpn_threshold = 0.5

    def __init__(self, model_path='frozen_inference_graph.pb'):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.333
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model_path = model_path
        self.d_graph = tf.Graph()
        with self.d_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.compat.v1.Session(config=config)
            # self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}

            # FOR RCNN final layer results:
            # self.tensor_dict_high = {}
            self.tensor_dict_low = {}
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict_low[key] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))
                    # self.tensor_dict_high[key] = (tf.compat.v1.get_default_graph()
                    #                     .get_tensor_by_name(tensor_name))
            # FOR RPN intermedia results
            # self.key_tensor_map = {
            #     "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
            #                             "/map/while/"
            #                             "MultiClassNonMaxSuppression/"
            #                             "Gather/Gather:0"),
            #     "RPN_score": ("BatchMultiClassNonMaxSuppression/"
            #                 "map/while/"
            #                 "MultiClassNonMaxSuppression"
            #                 "/Gather/Gather_2:0"),
            #     "Resized_shape": ("Preprocessor/map/while"
            #                     "/ResizeToRange/stack_1:0"),
            # }

            # for key, tensor_name in self.key_tensor_map.items():
            #     if tensor_name in all_tensor_names:
            #         self.tensor_dict_low[tensor_name] = (
            #             tf.compat.v1.get_default_graph()
            #             .get_tensor_by_name(tensor_name))

        self.logger.info("Object detector initialized")



        # self.logger = logging.getLogger("object_detector")
        # handler = logging.NullHandler()
        # self.logger.addHandler(handler)

        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        # config = ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.model_rpn_path = model_rpn_path
        # self.d_graph_rpn = tf.Graph()
        # with self.d_graph_rpn.as_default():
        #     od_graph_def = tf.compat.v1.GraphDef()
        #     with tf.io.gfile.GFile(self.model_rpn_path, 'rb') as fid:
        #         serialized_graph = fid.read()
        #         od_graph_def.ParseFromString(serialized_graph)
        #         tf.import_graph_def(od_graph_def, name='')
        #     self.session_rpn = tf.compat.v1.Session(config=config)
        
        #     ops = tf.compat.v1.get_default_graph().get_operations()
        #     all_tensor_names = {output.name for op in ops
        #                         for output in op.outputs}

        #     # FOR RCNN final layer results:
        #     self.tensor_dict_rpn = {}
        #     # self.tensor_dict_infer = {}
        #     # for key in [
        #     #         'num_detections', 'detection_boxes',
        #     #         'detection_scores', 'detection_classes',
        #     #         'detection_masks'
        #     # ]:
        #     #     tensor_name = key + ':0'
        #     #     if tensor_name in all_tensor_names:
        #     #         self.tensor_dict_infer[key] = (tf.compat.v1.get_default_graph()
        #     #                             .get_tensor_by_name(tensor_name))
        #             # self.tensor_dict_high[key] = (tf.compat.v1.get_default_graph()
        #             #                     .get_tensor_by_name(tensor_name))
        #     # FOR RPN intermedia results
        #     self.key_tensor_map = {
        #         "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
        #                                 "/map/while/"
        #                                 "MultiClassNonMaxSuppression/"
        #                                 "Gather/Gather:0"),
        #         "RPN_score": ("BatchMultiClassNonMaxSuppression/"
        #                     "map/while/"
        #                     "MultiClassNonMaxSuppression"
        #                     "/Gather/Gather_2:0"),
        #         "Resized_shape": ("Preprocessor/map/while"
        #                         "/ResizeToRange/stack_1:0"),
        #     }

        #     for key, tensor_name in self.key_tensor_map.items():
        #         if tensor_name in all_tensor_names:
        #             self.tensor_dict_rpn[tensor_name] = (
        #                 tf.compat.v1.get_default_graph()
        #                 .get_tensor_by_name(tensor_name))

        # self.logger.info("Object detector initialized")

    def run_inference_for_single_image(self, image, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}

            # FOR RCNN final layer results:
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            # FOR RPN intermedia results
            key_tensor_map = {
                "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
                                          "/map/while/"
                                          "MultiClassNonMaxSuppression/"
                                          "Gather/Gather:0"),
                "RPN_score": ("BatchMultiClassNonMaxSuppression/"
                              "map/while/"
                              "MultiClassNonMaxSuppression"
                              "/Gather/Gather_2:0"),
                "Resized_shape": ("Preprocessor/map/while"
                                  "/ResizeToRange/stack_1:0"),
            }

            for key, tensor_name in key_tensor_map.items():
                if tensor_name in all_tensor_names:
                    tensor_dict[tensor_name] = (
                        tf.compat.v1.get_default_graph()
                        .get_tensor_by_name(tensor_name))

            image_tensor = (tf.compat.v1.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            # Run inference
            feed_dict = {image_tensor: np.expand_dims(image, 0)}
            infer_start = perf_counter_s()
            output_dict = self.session.run(tensor_dict,
                                           feed_dict=feed_dict)
            infer_end = perf_counter_s() - infer_start

            # FOR RPN intermedia results
            w = output_dict[key_tensor_map['Resized_shape']][1]
            h = output_dict[key_tensor_map['Resized_shape']][0]
            input_shape_array = np.array([h, w, h, w])
            output_dict['RPN_box_normalized'] = output_dict[key_tensor_map[
                'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]
            output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]

            # appNum = os.popen('pwd').read()
            # appNum = int(appNum[appNum.rfind("app")+3])
            # writeResult(appNum, output_dict, "inferDebug")

            # FOR RCNN final layer results
            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'][0])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'][0])
        return output_dict, infer_end
    
    # def run_inference_for_tensor(self, images, graph):
    #     with self.d_graph.as_default():
    #         # Get handles to input and output tensors
    #         ops = tf.compat.v1.get_default_graph().get_operations()
    #         all_tensor_names = {output.name for op in ops
    #                             for output in op.outputs}

    #         # FOR RCNN final layer results:
    #         tensor_dict = {}
    #         for key in [
    #                 'num_detections', 'detection_boxes',
    #                 'detection_scores', 'detection_classes',
    #                 'detection_masks'
    #         ]:
    #             tensor_name = key + ':0'
    #             if tensor_name in all_tensor_names:
    #                 tensor_dict[key] = (tf.compat.v1.get_default_graph()
    #                                     .get_tensor_by_name(tensor_name))

    #         # FOR RPN intermedia results
    #         key_tensor_map = {
    #             "RPN_box_no_normalized": ("BatchMultiClassNonMaxSuppression"
    #                                       "/map/while/"
    #                                       "MultiClassNonMaxSuppression/"
    #                                       "Gather/Gather:0"),
    #             "RPN_score": ("BatchMultiClassNonMaxSuppression/"
    #                           "map/while/"
    #                           "MultiClassNonMaxSuppression"
    #                           "/Gather/Gather_2:0"),
    #             "Resized_shape": ("Preprocessor/map/while"
    #                               "/ResizeToRange/stack_1:0"),
    #         }

    #         for key, tensor_name in key_tensor_map.items():
    #             if tensor_name in all_tensor_names:
    #                 tensor_dict[tensor_name] = (
    #                     tf.compat.v1.get_default_graph()
    #                     .get_tensor_by_name(tensor_name))

    #         image_tensor = (tf.compat.v1.get_default_graph()
    #                         .get_tensor_by_name('image_tensor:0'))
    #         batch_size = images.shape[0]
    #         # Run inference
    #         feed_dict = {image_tensor: for }
    #         infer_start = perf_counter_s()
    #         output_dict = self.session.run(tensor_dict,
    #                                        feed_dict=feed_dict)
    #         infer_end = perf_counter_s() - infer_start

    #         # FOR RPN intermedia results
    #         w = output_dict[key_tensor_map['Resized_shape']][1]
    #         h = output_dict[key_tensor_map['Resized_shape']][0]
    #         input_shape_array = np.array([h, w, h, w])
    #         output_dict['RPN_box_normalized'] = output_dict[key_tensor_map[
    #             'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]
    #         output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]

    #         # FOR RCNN final layer results
    #         # all outputs are float32 numpy arrays,
    #         # so convert types as appropriate
    #         output_dict['num_detections'] = int(
    #             output_dict['num_detections'][0])
    #         output_dict['detection_classes'] = output_dict[
    #             'detection_classes'][0].astype(np.uint8)
    #         output_dict['detection_boxes'] = (
    #             output_dict['detection_boxes'][0])
    #         output_dict['detection_scores'] = (
    #             output_dict['detection_scores'][0])
    #     return output_dict, infer_end

    def run_inference_for_multiple_images(self, images, graph, second_phase):
        infer_end = 0
        output_dict = {}
        tensor_dict = None
        # RPN Generation
        # if not second_phase:
        #     with self.d_graph_rpn.as_default():
        #         image_tensor = (tf.compat.v1.get_default_graph()
        #                 .get_tensor_by_name('image_tensor:0'))
        #         for image in images:
        #             feed_dict = {image_tensor: np.expand_dims(image, axis=0)}
        #             tensor_dict = self.tensor_dict_rpn
        #             infer_and_process_start = perf_counter_s()
        #             temp_dict = self.session_rpn.run(tensor_dict,
        #                                     feed_dict=feed_dict)
            
        #             infer_end += perf_counter_s() - infer_and_process_start
        #             if len(output_dict) == 0: # the first image
        #                 output_dict = temp_dict
        #                 w = output_dict[self.key_tensor_map['Resized_shape']][1]
        #                 h = output_dict[self.key_tensor_map['Resized_shape']][0]
        #                 input_shape_array = np.array([h, w, h, w])
        #                 output_dict['RPN_box_normalized'] = [output_dict[self.key_tensor_map[
        #                     'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]]
        #                 output_dict['RPN_score'] = [output_dict[self.key_tensor_map['RPN_score']]]
        #             else:
        #                 # for key in [
        #                 # 'num_detections', 'detection_boxes',
        #                 # 'detection_scores', 'detection_classes']:
        #                 #     output_dict[key] = np.append(output_dict[key], temp_dict[key], axis=0)
        #                 w = temp_dict[self.key_tensor_map['Resized_shape']][1]
        #                 h = temp_dict[self.key_tensor_map['Resized_shape']][0]
        #                 input_shape_array = np.array([h, w, h, w])
        #                 output_dict['RPN_box_normalized'].append((temp_dict[self.key_tensor_map[
        #                     'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]))
        #                 output_dict['RPN_score'].append(temp_dict[self.key_tensor_map['RPN_score']])
        
        with self.d_graph.as_default():
            output_dict = {}
            tensor_dict = None
            image_tensor = (tf.compat.v1.get_default_graph()
                        .get_tensor_by_name('image_tensor:0'))
            # appNum = os.popen('pwd').read()
            # appNum = int(appNum[appNum.rfind("app")+3]) 
            # writeResult(appNum, image_tensor, "sanityCheck")
            
            # Get handles to input and output tensors
            # Run inference

            # start of experiment
            # if False:
            #     # generate rpn first
            #     for image in images:
            #         feed_dict = {image_tensor: np.expand_dims(image, axis=0)}
            #         tensor_dict = self.tensor_dict_rpn
            #         infer_and_process_start = perf_counter_s()
            #         temp_dict = self.session.run(tensor_dict,
            #                                 feed_dict=feed_dict)
            
            #         infer_end += perf_counter_s() - infer_and_process_start
            #         if len(output_dict) == 0: # the first image
            #             output_dict = temp_dict
            #             w = output_dict[self.key_tensor_map['Resized_shape']][1]
            #             h = output_dict[self.key_tensor_map['Resized_shape']][0]
            #             input_shape_array = np.array([h, w, h, w])
            #             output_dict['RPN_box_normalized'] = [output_dict[self.key_tensor_map[
            #                 'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]]
            #             output_dict['RPN_score'] = [output_dict[self.key_tensor_map['RPN_score']]]
            #         else:
            #             # for key in [
            #             # 'num_detections', 'detection_boxes',
            #             # 'detection_scores', 'detection_classes']:
            #             #     output_dict[key] = np.append(output_dict[key], temp_dict[key], axis=0)
            #             w = temp_dict[self.key_tensor_map['Resized_shape']][1]
            #             h = temp_dict[self.key_tensor_map['Resized_shape']][0]
            #             input_shape_array = np.array([h, w, h, w])
            #             output_dict['RPN_box_normalized'].append((temp_dict[self.key_tensor_map[
            #                 'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]))
            #             output_dict['RPN_score'].append(temp_dict[self.key_tensor_map['RPN_score']])
            # ## end of experiment
            
            # # infer batching
            # img = images
            # for i in range(4):
            #     images += images
            feed_dict = {image_tensor: np.array(images)}
            tensor_dict = self.tensor_dict_low
            # infer_and_process_start = perf_counter_s()
            # temp_dict = self.session.run(tensor_dict,
            #                             feed_dict=feed_dict)
            # infer_end += perf_counter_s() - infer_and_process_start 
            infer_and_process_start = perf_counter_s()
            temp_dict = self.session.run(tensor_dict,
                                        feed_dict=feed_dict)
            # temp_dict = self.session.run(tensor_dict,
            #                             feed_dict=feed_dict)
            # temp_dict = self.session.run(tensor_dict,
            #                             feed_dict=feed_dict)
            # temp_dict = self.session.run(tensor_dict,
            #                             feed_dict=feed_dict)
            infer_end = perf_counter_s() - infer_and_process_start         

            output_dict = dict(output_dict, **temp_dict)
            # writeResult(appNum, infer_end_temp, "delayMulti") 

            # Normal inference
            # for image in images:
            #     if second_phase:
            #         feed_dict = {image_tensor: np.array(images)}
            #         tensor_dict = self.tensor_dict_high
            #     else:
            #         feed_dict = {image_tensor: np.expand_dims(image, axis=0)}
            #         tensor_dict = self.tensor_dict_low
            #     # feed_dict = {image_tensor: np.array(images)}
            #     infer_and_process_start = perf_counter_s()
            #     temp_dict = self.session.run(tensor_dict,
            #                                 feed_dict=feed_dict)
            
            #     infer_end += perf_counter_s() - infer_and_process_start

            #     if second_phase:
            #         output_dict = temp_dict
            #         break
            #     if len(output_dict) == 0: # the first image
            #         output_dict = temp_dict
            #         # w = output_dict[self.key_tensor_map['Resized_shape']][1]
            #         # h = output_dict[self.key_tensor_map['Resized_shape']][0]
            #         # input_shape_array = np.array([h, w, h, w])
            #         # output_dict['RPN_box_normalized'] = [output_dict[self.key_tensor_map[
            #         #     'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]]
            #         # output_dict['RPN_score'] = [output_dict[self.key_tensor_map['RPN_score']]]
            #     else:
            #         pass
            #         for key in [
            #         'num_detections', 'detection_boxes',
            #         'detection_scores', 'detection_classes']:
            #             output_dict[key] = np.append(output_dict[key], temp_dict[key], axis=0)
            #         # w = temp_dict[self.key_tensor_map['Resized_shape']][1]
            #         # h = temp_dict[self.key_tensor_map['Resized_shape']][0]
            #         # input_shape_array = np.array([h, w, h, w])
            #         # output_dict['RPN_box_normalized'].append((temp_dict[self.key_tensor_map[
            #         #     'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]))
            #         # output_dict['RPN_score'].append(temp_dict[self.key_tensor_map['RPN_score']])

            # writeResult(appNum, output_dict, "inferDebug")
            # FOR RPN intermedia results
            

            # FOR RCNN final layer results
            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = output_dict[
                'num_detections'].astype('int32')
            output_dict['detection_classes'] = output_dict[
                'detection_classes'].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'])
        # writeResult(appNum, str(perf_counter_s() - infer_and_process_start), "inferDebugging")
        return output_dict, infer_end
    
    def run_inference_for_multiple_images_high(self, images, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}

            # FOR RCNN final layer results:
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            image_tensor = (tf.compat.v1.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            # Run inference

            # imgs = np.array([image, image])
            # feed_dict = {image_tensor: np.expand_dims(image, 0)}
            # frames = images.shape[0]
            # imgs = np.append(images, images, axis=0)
            # imgs = np.append(imgs, images, axis=0)
            images = np.array(images)
            feed_dict = {image_tensor: images}
            infer_start = perf_counter_s()
            output_dict = self.session.run(tensor_dict,
                                           feed_dict=feed_dict)
            # appNum = os.popen('pwd').read()
            # appNum = int(appNum[appNum.rfind("app")+3])

            infer_end = perf_counter_s() - infer_start
            # writeResult(appNum, output_dict, "secondInferDebug")
            # FOR RPN intermedia results
            # w = output_dict[key_tensor_map['Resized_shape']][1]
            # h = output_dict[key_tensor_map['Resized_shape']][0]
            # input_shape_array = np.array([h, w, h, w])
            # output_dict['RPN_box_normalized'] = output_dict[key_tensor_map[
            #     'RPN_box_no_normalized']]/input_shape_array[np.newaxis, :]
            # output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]

            # FOR RCNN final layer results
            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = output_dict[
                'num_detections'].astype('int32')
            output_dict['detection_classes'] = output_dict[
                'detection_classes'].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'])
        return output_dict, infer_end

    def inferHigh(self, image_np):
        imgae_crops = image_np

        output_dict, infer_delay = self.run_inference_for_multiple_images_high(
            imgae_crops, self.d_graph)

        results = []
        for image in range(len(imgae_crops)):
        # for image in range(1):
            result_temp = []
            for i in range(len(output_dict['detection_boxes'][image])):
                object_class = output_dict['detection_classes'][image][i]
                relevant_class = False
                for k in Detector.classes.keys():
                    if object_class in Detector.classes[k]:
                        object_class = k
                        relevant_class = True
                        break
                if not relevant_class:
                    continue

                ymin, xmin, ymax, xmax = output_dict['detection_boxes'][image][i]
                confidence = output_dict['detection_scores'][image][i]
                box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
                result_temp.append((object_class, confidence, box_tuple))
            results.append(result_temp)

        # Get RPN regions along with classification results
        # rpn results array will have (class, (xmin, xmax, ymin, ymax)) typles
        results_rpn = []
        # for idx_region, region in enumerate(output_dict['RPN_box_normalized']):
        #     x = region[1]
        #     y = region[0]
        #     w = region[3] - region[1]
        #     h = region[2] - region[0]
        #     conf = output_dict['RPN_score'][idx_region]
        #     if conf < Detector.rpn_threshold or w * h == 0.0 or w * h > 0.04:
        #         continue
        #     results_rpn.append(("object", conf, (x, y, w, h)))

        return results, results_rpn, infer_delay

    def infer(self, image_np, second_phase, executor):
        imgae_crops = image_np

        # # this will be used as the implementation for batching processing
        output_dict, infer_delay = self.run_inference_for_multiple_images(
            imgae_crops, self.d_graph, second_phase)

        # this output_dict contains both final layer results and RPN results
        # output_dict, infer_delay = self.run_inference_for_single_image(
        #     imgae_crops, self.d_graph)

        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        results_rpn = []

        size = len(imgae_crops)

        if not second_phase:
            results, results_rpn = self.process_first_phase(output_dict, size, executor)
        else:
            results, results_rpn = self.process_second_phase(output_dict, size, executor)
        
        # for i in range(len(output_dict['detection_boxes'])):
        #     object_class = output_dict['detection_classes'][i]
        #     relevant_class = False
        #     for k in Detector.classes.keys():
        #         if object_class in Detector.classes[k]:
        #             object_class = k
        #             relevant_class = True
        #             break
        #     if not relevant_class:
        #         continue

        #     ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
        #     confidence = output_dict['detection_scores'][i]
        #     box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
        #     results.append((object_class, confidence, box_tuple))

        # Get RPN regions along with classification results
        # rpn results array will have (class, (xmin, xmax, ymin, ymax)) typles
        
        # for idx_region, region in enumerate(output_dict['RPN_box_normalized']):
        #     x = region[1]
        #     y = region[0]
        #     w = region[3] - region[1]
        #     h = region[2] - region[0]
        #     conf = output_dict['RPN_score'][idx_region]
        #     if conf < Detector.rpn_threshold or w * h == 0.0 or w * h > 0.04:
        #         continue
        #     results_rpn.append(("object", conf, (x, y, w, h)))

        return results, results_rpn, infer_delay
    
    def process_first_phase(self, output_dict, num_of_frames, executor):
        results = []
        results_rpn = []
        output_dict_list = [output_dict for i in range(num_of_frames)]
        nums = [i for i in range(num_of_frames)]
        detectors = [Detector for i in range(num_of_frames)]
        argsZipped = list(zip(output_dict_list, nums, detectors))
        multi_proc = executor.starmap(first_phase_sub, argsZipped)
        for result in multi_proc:
            results.append(result[0])
            results_rpn.append(result[1])
        # for image in range(num_of_frames):
        # # for image in range(1):
        #     result_temp = []
        #     for i in range(len(output_dict['detection_boxes'][image])):
        #         object_class = output_dict['detection_classes'][image][i]
        #         relevant_class = False
        #         for k in Detector.classes.keys():
        #             if object_class in Detector.classes[k]:
        #                 object_class = k
        #                 relevant_class = True
        #                 break
        #         if not relevant_class:
        #             continue

        #         ymin, xmin, ymax, xmax = output_dict['detection_boxes'][image][i]
        #         confidence = output_dict['detection_scores'][image][i]
        #         box_tuple = (xmin, ymin, xmax - xmin, ysmax - ymin)
        #         result_temp.append((object_class, confidence, box_tuple))
        #     results.append(result_temp)

        #     result_temp = []
        #     for idx_region, region in enumerate(output_dict['RPN_box_normalized'][image]):
        #         x = region[1]
        #         y = region[0]
        #         w = region[3] - region[1]
        #         h = region[2] - region[0]
        #         conf = output_dict['RPN_score'][image][idx_region]
        #         if conf < Detector.rpn_threshold or w * h == 0.0 or w * h > 0.04:
        #             continue
        #         result_temp.append(("object", conf, (x, y, w, h)))
        #     results_rpn.append(result_temp)
        return results, results_rpn
    
    def process_second_phase(self, output_dict, num_of_frames, executor):
        # appNum = os.popen('pwd').read()
        # appNum = int(appNum[appNum.rfind("app")+3])
        # start_sec = perf_counter_s()
        results = []
        results_rpn = []
        output_dict_list = [output_dict for i in range(num_of_frames)]
        nums = [i for i in range(num_of_frames)]
        detectors = [Detector for i in range(num_of_frames)]
        argsZipped = list(zip(output_dict_list, nums, detectors))
        multi_proc = executor.starmap(second_phase_sub, argsZipped)
        for result in multi_proc:
            results.append(result[0])
            results_rpn.append(result[1])
        # writeResult(appNum, perf_counter_s()-start_sec, "secPhase")
        
        # for image in range(num_of_frames):
        # # for image in range(1):
        #     start_sec = perf_counter_s()
        #     result_temp = []
        #     for i in range(len(output_dict['detection_boxes'][image])):
        #         object_class = output_dict['detection_classes'][image][i]
        #         relevant_class = False
        #         for k in Detector.classes.keys():
        #             if object_class in Detector.classes[k]:
        #                 object_class = k
        #                 relevant_class = True
        #                 break
        #         if not relevant_class:
        #             continue

        #         ymin, xmin, ymax, xmax = output_dict['detection_boxes'][image][i]
        #         confidence = output_dict['detection_scores'][image][i]
        #         box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
        #         result_temp.append((object_class, confidence, box_tuple))
        #     results.append(result_temp)
        #     writeResult(appNum, perf_counter_s()-start_sec, "secPhase")
        
        return results, results_rpn
