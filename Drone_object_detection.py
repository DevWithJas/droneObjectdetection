import cv2
import cvzone
from djitellopy import tello 

thres = 0.6
nmsThres = 0.2

cap = cv2.VideoCapture(0)
cap.set(3,648)
cap.set(4,480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

print(classNames)
configpath = "J:\Machine Learing Projects\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath = "J:\Machine Learing Projects\frozen_inference_graph.pb"

net = cv2.dnn_ClassificationModel(weightPath,configpath)
net.setInputSize(320,320)
net.setinputScale(1.0/127.5)
net.setInputMean(12.5, 127.5 ,127.5)
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print(me.get_battery)
me.streamoff()
me.stream_on()

while True:
    success, img = cap.read()
    img = me.get_frame_read().frame
    classIds,confs,bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flattem, bbox):
            cvzone.cornerRect(img, box,)
            cv2.putText(img, f'{classNames[classId-1].upper()} {round(conf*100,2)}', (box[0]+ 10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1,(0,255,0),2)

    except:
        pass


    cv2.imshow("image", img)
    cv2.waitKey(1)


    