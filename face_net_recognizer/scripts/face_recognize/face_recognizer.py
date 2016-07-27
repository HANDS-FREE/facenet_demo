"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

import importlib
import numpy as np
import os

import cv2
import tensorflow as tf

import facenet
import align_dlib
import models.nn4 as network

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_file', '/home/air/workSpace/src/face_net_recognizer/scripts/face_recognize/model_data/model.ckpt-500000',
                           """File containing the model parameters as well as the model metagraph (with extension '.meta')""")
tf.app.flags.DEFINE_string('dlib_face_predictor', '/home/air/workSpace/src/face_net_recognizer/scripts/face_recognize/model_data/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_string('image1', '/home/air/datas/set/set1/ba.tif', """First image to compare.""")
tf.app.flags.DEFINE_string('image2', '/home/air/caffeTrain/1454769260902.jpg', """Second image to compare.""")


tf.app.flags.DEFINE_string('pool_type', 'MAX',
                           """The type of pooling to use for some of the inception layers {'MAX', 'L2'}.""")
tf.app.flags.DEFINE_boolean('use_lrn', False,
                            """Enables Local Response Normalization after the first layers of the inception network.""")
tf.app.flags.DEFINE_float('keep_probability', 1.0,
                          """Keep probability of dropout for the fully connected layer(s).""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")


class face_recognizer():
    def __init__(self):
        self.__align = align_dlib.AlignDlib(os.path.expanduser(FLAGS.dlib_face_predictor))
        self.__landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

        with tf.Graph().as_default():
            self.__sess = tf.Session()

            self.__images_placeholder = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3),name='input')
            self.__phase_train_placeholder  = tf.placeholder(tf.bool, name='phase_train')
            self.__embeddings = network.inference(self.__images_placeholder, FLAGS.pool_type, FLAGS.use_lrn,
                                           FLAGS.keep_probability, phase_train=self.__phase_train_placeholder)
            self.__image_size = int(self.__images_placeholder.get_shape()[1])

            self.__saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
            self.__saver.restore(self.__sess, FLAGS.model_file)
        return

    def canculate(self,image_paths):
        # Run forward pass to calculate embeddings
        images,success_flag = self.__load_and_align_data(image_paths, self.__image_size, self.__align, self.__landmarkIndices)
        if success_flag:
            feed_dict = {self.__images_placeholder: images, self.__phase_train_placeholder: False}
            emb = self.__sess.run(self.__embeddings, feed_dict=feed_dict)
            return emb
        else:
            return None

    def __load_and_align_data(self,image_paths, image_size, align, landmarkIndices):
        nrof_samples = len(image_paths)
        img_list = [None] * nrof_samples
        for i in xrange(nrof_samples):
            # img = misc.imread(image_paths[i])
            img = cv2.imread(image_paths[i])
            aligned = align.align(image_size, img, landmarkIndices=landmarkIndices, skipMulti=False)
            if aligned==None:
                return None,False
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened

        images = np.asarray(img_list)
        return images,True

    def set_inside_templates(self,templates_name):
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
            aligned = self.__align.align(self.__image_size, img, landmarkIndices=self.__landmarkIndices, skipMulti=False)
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
        self.__templates_number = 0
        self.__templates_label = []
        self.__templates_emb = []

    def compare_with_templates(self,images):
        images_number = len(images)
        img_list = [None]*images_number
        for i in xrange(images_number):
            aligned = self.__align.align(self.__image_size, images[i], landmarkIndices=self.__landmarkIndices, skipMulti=False)
            if aligned == None:
                return None,None
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
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
            return all_label,all_belief_list
        else :
            return None,None







