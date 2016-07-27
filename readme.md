##**face_net_recognizer**
its a ros package depend on **tensorflow** and **dlib**
tensorflow -- https://www.tensorflow.org/
dlib -- http://dlib.net/

the face_net part is based on https://github.com/davidsandberg/facenet
in <face_net_recognizer>\scripts\face_recognize\model_data
has two file
model.ckpt-500000 which is a tensorflow checkpoint file  
The data has been pre-processed as described on the OpenFace web page (https://cmusatyalab.github.io/openface/models-and-accuracies/), i.e. using ./util/align-dlib.py data/lfw/raw align outerEyesAndNose data/lfw/dlib-affine-sz:96 --size 96 --fallbackLfw data/lfw/deepfunneled

the shape_predictor_68_face_landmarks.dat
is a face_landmark file based on dlib


in the <face_net_recognizer>\scripts\face_recognize\templates
we have two sample people and a describe_csv file

change them to what ever you want to use in the face_recognitio :)

facenet is based on google's paperFaceNet: A Unified Embedding for Face Recognition and Clustering
it use a novel method mot formillar with the classfiermodel
so only one templates and on training would perfomes well :)