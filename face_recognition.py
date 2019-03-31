import cv2 
import numpy as np
import os
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,3)
   #for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    return faces,gray

#video_capture=cv2.VideoCapture(0)

#while True:
  #  _,frame=video_capture.read()
   
   # canvas,gray=detect(frame) 
   # cv2.imshow('?N',canvas)
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    Video_capture.release()
#        cv2.destroyAllWindows()  

def lebels_for_traningData(directory):
    faces=[]
    faceId=[]
    for (path,subdir,filenames) in os.walk(directory):
        for(filename) in filenames:
            if filename.startswith("."):
                print("skipping",filename)
                continue
            Id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            test_img=cv2.imread(img_path)
            rect,gray=detect(test_img)
            if len(rect)>0:
              (x,y,w,h)=rect[0]
            roi_gray=gray[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceId.append(int(Id))
    return faces,faceId
def train_Classifier(faces,faceId):
     face_recogn=cv2.face.LBPHFaceRecognizer_create()
     face_recogn.train(faces,np.array(faceId))
     return face_recogn
 
def draw_rect(test_img,faces):
    
     for (x,y,w,h) in faces:
       cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),2)
   
def put_text(test_img,text,x,y):
   cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1)    
   
video_capture=cv2.VideoCapture(0)
while True:
    _,test_img=video_capture.read()
#test_img=cv2.imread('/mnt/7E5CD04E5CD00337/opn cv/bhajssjd.jpeg')
    #test_img=cv2.imread(frame)
    faces,faceId=lebels_for_traningData('/mnt/7E5CD04E5CD00337/opn cv/traning data')
    face_detected,gray=detect(test_img)

    face_recogn=train_Classifier(faces,faceId)
    name={0:"Salman",1:"Priyanka Chopda",2:"Bill Gates"} 
    for face in face_detected:
        (x,y,w,h)=face
        roi_gray=gray[y:y+w,x:x+h]
        lebel,confidence=face_recogn.predict(roi_gray)
        predicted_name=name[lebel]
        print("confidence : ",confidence)
        if confidence>120:
              put_text(test_img,predicted_name,x-(x/2),y)
    draw_rect(test_img,face_detected)
    resized_img=cv2.resize(test_img,(1000,700))    
    cv2.imshow("recong",test_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows    
    
    
    
    
    
    
    
    
    
    