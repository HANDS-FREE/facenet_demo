#!/usr/bin/env python
import face_recognize.face_recognizer as recognizer
import cv2
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import face_recognize.face_recognizer_ros_node as recognizer_ros

if __name__ == '__main__':
    # r = recognizer.face_recognizer()
    #
    #
    # f = open("/home/air/datas/testFileld.txt")
    # lines= f.readlines()
    # success = r.set_inside_templates(lines)
    # if not success:
    #     print "fail"
    #
    # f = open("/home/air/datas/test.txt")
    # count = 0
    #
    # loop = 110
    # size = 1
    #
    # for i in range(0,loop):
    #     path_sub = [None]*size
    #     label_sub = [None]*size
    #     img_sub = [None]*size
    #     for i in range(0,size):
    #         line = f.readline()
    #         path, label = line.split(';')
    #         path_sub[i] = path
    #         label_sub[i] = int(label)
    #         img = cv2.imread(path)
    #         img_sub[i] = img
    #
    #
    #     indicates,belief = r.compare_with_templates(img_sub)
    #
    #     print belief
    #
    #     print indicates
    #     print label_sub
    #     print "------------------"
    #     for i in range(0,size):
    #         if indicates[i] == label_sub[i]:
    #     #         print indicates[i] ,"------------",label_sub[i]
    #             count = count +1
    #
    #
    # print count
    #     # print b
    finle = open("/home/air/datas/mySmallTest/templates.txt")
    lines = finle.readlines()
    sub = recognizer_ros.subscribe_image_and_recognize_face(lines,"/usb_cam/image_raw")
    rospy.init_node('image_converter', anonymous=True)
    sub.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

