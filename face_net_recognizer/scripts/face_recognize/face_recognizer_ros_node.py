#!/usr/bin/env python
import sys
import roslib
import rospy
import numpy as np
import cv2
import face_recognizer
#from face_tracker.srv import *
from sensor_msgs.msg import Image as sensorImg
from cv_bridge import CvBridge,CvBridgeError
import message_filters
# from dqn_node.srv import frameScore_withIMG,frameScore_withIMGRequest,environment,environmentResponse
#from dqn_node.srv import environment, environmentResponse
import time

class rewardService:
    def __init__(self):
        self.bridge = CvBridge()
        self.service = rospy.Service('Environment',"<your message type>",self.rewardHandle)
        self.new = False
        return

    def rewardHandle(self,req):
        action = req.action
        image,reward,terminal=self.env.get_image_and_reward(action)
        rpy = environmentResponse()
        rpy.reward = reward
        rpy.terminal = terminal
        rpy.img = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.new = True
        return rpy

    def loop(self):
        while not rospy.is_shutdown():
            if self.new:
                self.env.data_log()
                self.new = False
            rospy.sleep(0.01)

class subscribe_image_and_recognize_face():
    def __init__(self,template_file_list,subscribe_topic_name):
        self.recognizer = face_recognizer.face_recognizer()
        if not self.recognizer.set_inside_templates(template_file_list):
            print "fail"
            exit()
        self.bridge = CvBridge()
        # message_filters.Subscriber()
        # self.image_sub = rospy.Subscriber(subscribe_topic_name, sensorImg, self.callback,queue_size=1,tcp_nodelay=True)
        '''pleas read here  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '''
        sub = message_filters.Subscriber(subscribe_topic_name, sensorImg)
        self.cache = message_filters.Cache(sub, 100)

    def run(self):
        while True:
            data = self.cache.getElemAfterTime(self.cache.getLastestTime())
            self.callback(data)

''''...........................................................'''


    def callback(self,data):
        time1 = time.clock()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        time2 = time.clock()

        labels,belifs = self.recognizer.compare_with_templates([cv_image])

        time3 = time.clock()

        if labels == None:
            cv2.putText(cv_image,"no face detected",(20,40),cv2.FONT_ITALIC,1,(0,0,255),2,4)
        else:
            label = labels[0]
            belief = np.asarray(belifs[0])
            belief = belief[np.argmax(belief)]
            text = "lbl="+str(label)+" pro="+str(belief)
            cv2.putText(cv_image,text,(20,40),cv2.FONT_ITALIC,1,(0,255,0),2,4)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        print time2-time1,time3-time2


