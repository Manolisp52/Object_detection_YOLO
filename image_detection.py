import cv2
import numpy as np
config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"
model=cv2.dnn_DetectionModel(frozen_model,config_file)
Labels=[]
file_name="labels.txt"
with open(file_name,"rt") as fpt:
    Labels= fpt.read().rstrip("\n").split("\n") #δημιουργείται η λίστα Labels με τα ονόματα των αντικειμένων που θα αναγνωρίζονται

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

def image_detection():     ## "διαβάζει" μια εικόνα και την επιστρέφει έχοντας κάνει αναγνώριση
    img=cv2.imread("car_image.jpg") #"διβάζει" την εικόνα
    ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,color=(0,255,0),thickness=2)
        cv2.putText(img,Labels[ClassInd-1].upper(),(boxes[0]+10,boxes[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("detected",img)  #δείχνει την εικόνα με την αναγνώριση
    cv2.imwrite('car_detected.jpg',img) #αποθηκέυει την εικόνα με την αναγνώριση

image_detection()

