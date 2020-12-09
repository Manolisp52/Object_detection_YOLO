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

def image_detection():##read an image and show detected image
    img=cv2.imread("car_image.jpg")
    ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,color=(0,255,0),thickness=2)
        cv2.putText(img,Labels[ClassInd-1].upper(),(boxes[0]+10,boxes[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("detected",img)
    cv2.imwrite('car_detected.jpg',img)

def webcam_detection():
    thres=0.5
    nms_threshold = 0.2
    cap = cv2.VideoCapture(0)
    while True:
        success,img = cap.read()
        classIds, confs, bbox = model.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            cv2.putText(img,Labels[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    cv2.waitKey(1)

# image_detection()
webcam_detection()

