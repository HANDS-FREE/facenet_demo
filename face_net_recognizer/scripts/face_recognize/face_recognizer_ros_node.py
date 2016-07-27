#!/usr/bin/env python
import sys
import roslib
import rospy
import numpy as np
import cv2
import face_recognizer
from sensor_msgs.msg import Image as sensorImg
from cv_bridge import CvBridge,CvBridgeError
import message_filters
from face_net_recognizer.srv import frame_beliefs,frame_beliefsResponse,frame_beliefsRequest
import thread

class build_service_and_recognize():
    def __init__(self,template_file_list_name,model_file,dlib_face_predictor):
        self.recognizer = face_recognizer.face_recognizer(model_file,dlib_face_predictor)
        if not self.recognizer.set_inside_templates(template_file_list_name):
            print "fail"
            exit()
        self.bridge = CvBridge()
        self.service = rospy.Service('frame_beliefs',frame_beliefs,self.process)
        return

    def process(self,req):
        img = req.img
        grd_truth = req.grdTruth

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        labels, belifs, bounding_box_list = self.recognizer.compare_with_templates([cv_image])

        if labels == None:
            cv2.putText(cv_image, "no face detected", (20, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, 4)
        else:
            bb = bounding_box_list[0]
            label = labels[0]
            belief = np.asarray(belifs[0])
            belief = belief[np.argmax(belief)]
            text = "lbl=" + str(label) + " pro=" + str(belief)
            cv2.putText(cv_image, text, (20, 40), cv2.FONT_ITALIC, 1, (0, 255, 0), 2, 4)
            bounding_box_list = list(bounding_box_list)
            cv2.rectangle(cv_image, bb[0], bb[1], (255, 255, 0), 4)
        cv2.imshow("Image window", cv_image)
        key = cv2.waitKey(3)
        if key == 'q':
            exit()

        rpy = frame_beliefsResponse()
        rpy.score = belifs[0]
        return rpy

    def loop(self):
        while not rospy.is_shutdown():
            if self.new:
                self.env.data_log()
                self.new = False
            rospy.sleep(0.01)

class subscribe_image_and_recognize():
    def __init__(self,template_file_list_name,subscribe_topic_name,model_file,dlib_face_predictor):
        self.recognizer = face_recognizer.face_recognizer(model_file,dlib_face_predictor)
        if not self.recognizer.set_inside_templates(template_file_list_name):
            print "fail"
            exit()
        self.bridge = CvBridge()

        sub = message_filters.Subscriber(subscribe_topic_name, sensorImg)
        self.cache = message_filters.Cache(sub, 100)
        thread.start_new_thread(self.run,())

    def run(self):
        flag = True
        while flag:
            data = self.cache.getElemAfterTime(self.cache.getLastestTime())
            if data == None:
                pass
            else:
                flag = self.callback(data)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if cv_image == None:
            return True

        labels,belifs,bounding_box_list = self.recognizer.compare_with_templates([cv_image])

        if labels == None:
            cv2.putText(cv_image,"no face detected",(20,40),cv2.FONT_ITALIC,1,(0,0,255),2,4)
        else:
            bb = bounding_box_list[0]
            label = labels[0]
            belief = np.asarray(belifs[0])
            belief = belief[np.argmax(belief)]
            text = "lbl="+str(label)+" pro="+str(belief)
            cv2.putText(cv_image,text,(20,40),cv2.FONT_ITALIC,1,(0,255,0),2,4)
            cv2.rectangle(cv_image,bb[0],bb[1],(255,255,0),4)
        cv2.imshow("Image window", cv_image)
        key = cv2.waitKey(3)
        if key == 'q':
            return False

        return True


