#!/usr/bin/env python
import face_recognize.face_recognizer as recognizer
import cv2
import sys
import rospy
import face_recognize.face_recognizer_ros_node as recognizer_ros
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_file',
                           '/home/air/workSpace/src/face_net_recognizer/scripts/face_recognize/model_data/model.ckpt-500000',
                           """File containing the model parameters as well as the model metagraph (with extension '.meta')""")
tf.app.flags.DEFINE_string('dlib_face_predictor',
                           '/home/air/workSpace/src/face_net_recognizer/scripts/face_recognize/model_data/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_string('template_file_list_name',
                           '/home/air/workSpace/src/face_net_recognizer/scripts/face_recognize/templates/templates_describe_csv.txt',
                           """File containing the scv_templates describers""")
tf.app.flags.DEFINE_string('subscribe_topic_name',
                           '/usb_cam/image_raw',
                           """the subscribe topic's name""")

if __name__ == '__main__':
    sub = recognizer_ros.subscribe_image_and_recognize(FLAGS.template_file_list_name,FLAGS.subscribe_topic_name,FLAGS.model_file,FLAGS.dlib_face_predictor)
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

