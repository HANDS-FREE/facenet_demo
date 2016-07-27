"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

import importlib
import numpy as np
import os

import cv2
import tensorflow as tf

import facenet
import align_dlib
import models.nn4 as network

pool_type = 'MAX' #MAX  or  L2
use_lrn = False #Enables Local Response Normalization after the first layers of the inception network
keep_probability = 1.0 #Keep probability of dropout for the fully connected layer(s)
image_size = 96 #Image size (height, width) in pixels


class face_recognizer():
    def __init__(self,model_file,dlib_face_predictor):
        '''
        inital function

        :param model_file:the path point to chekpoint file (now using the model.ckpt-500000)
        :type basestring
        :param dlib_face_predictor: the path point to shape_predictor_68_face_landmarks.dat
        :type basestring
        '''
        self.__align = align_dlib.AlignDlib(os.path.expanduser(dlib_face_predictor))
        self.__landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

        with tf.Graph().as_default():
            self.__sess = tf.Session()

            self.__images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3),name='input')
            self.__phase_train_placeholder  = tf.placeholder(tf.bool, name='phase_train')
            self.__embeddings = network.inference(self.__images_placeholder, pool_type, use_lrn,
                                           keep_probability, phase_train=self.__phase_train_placeholder)
            self.__image_size = int(self.__images_placeholder.get_shape()[1])

            self.__saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
            self.__saver.restore(self.__sess, model_file)
        return

    def set_inside_templates(self,templates_file_name):
        '''
        build the inside template

        :param templates_file_name:path point to the templates_describe scv file
         the format wouldlike this:
            absolutely_path_to_image_1;label_1
            absolutely_path_to_image_2;label_2
            absolutely_path_to_image_3;label_3
            ...
        :type basestring
        :return: True if it success loaded the templates, or False
        '''
        templates_file = open(templates_file_name)
        templates_name = templates_file.readlines()
        self.clear_templates()
        self.__templates_number = len(templates_name)
        self.__templates_label = [None]*self.__templates_number
        img_list = [None] * self.__templates_number
        for i in xrange(self.__templates_number):
            # img = misc.imread(image_paths[i])
            line = templates_name[i].strip('\n')
            name,label = line.split(';')
            self.__templates_label[i]=int(label)
            img = cv2.imread(name)
            print name,"is reading ..."
            aligned,_ = self.__align.align(self.__image_size, img, landmarkIndices=self.__landmarkIndices, skipMulti=False)
            if aligned == None:
                return False
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
        # self.__templates = img_list
        img_array = np.asarray(img_list)
        feed_dict = {self.__images_placeholder: img_array, self.__phase_train_placeholder: False}
        emb = self.__sess.run(self.__embeddings, feed_dict=feed_dict)
        self.__templates_emb = emb
        return True

    def clear_templates(self):
        '''
        clear all the in_build templates
        '''
        self.__templates_number = 0
        self.__templates_label = []
        self.__templates_emb = []

    def compare_with_templates(self,images):
        '''
        compare input images with in_build templates

        :param images: images list which want to compare with in_build templates
        :type: a list of opencv_image
        :return:
            all_label:a list of label corresponding to the order of input image lists
            all_belief_list:a list of belief_list,
            bounding_box_list:a list of bounding box show where the face is

            None, None, None if angthing wrong in process
        '''
        images_number = len(images)
        img_list = [None]*images_number
        bounding_box_list = [None]*images_number
        for i in xrange(images_number):
            print i
            aligned,bounding_box = self.__align.align(self.__image_size, images[i], landmarkIndices=self.__landmarkIndices, skipMulti=False)
            if aligned == None:
                return None,None,None
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
            bounding_box_list[i] = bounding_box
        img_array = np.asarray(img_list)
        feed_dict = {self.__images_placeholder: img_array, self.__phase_train_placeholder: False}
        emb = self.__sess.run(self.__embeddings, feed_dict=feed_dict)

        if not emb == None:
            all_belief_list = [None]*images_number
            all_label = [None]*images_number
            for i in xrange(images_number):
                dis_list = [None]*self.__templates_number
                for j in range(0, self.__templates_number):
                    dist = np.sqrt(np.mean(np.square(np.subtract(self.__templates_emb[j, :], emb[i, :]))))
                    dis_list[j] = dist
                dis_array = np.array(dis_list)
                indicate = np.argmin(dis_array)

                dis_belief = np.exp(-3*dis_array)
                label = self.__templates_label[indicate]
                all_belief_list[i] = dis_belief
                all_label[i] = label
            return all_label,all_belief_list,bounding_box_list
        else :
            return None,None,None







