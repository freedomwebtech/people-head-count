import cv2
import numpy as np
from tracker import*
import cvzone
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=160)

# Open a video capture
video_capture = cv2.VideoCapture('headcount2.mp4')
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
tracker=Tracker()
cy1=222
cy2=247
offset=6
going_in={}
counter1=[]
going_out={}
counter2=[]

def denoise_mask(mask):
    kernel = np.ones((5, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    denoised_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return denoised_mask

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,500))
    # Apply background subtraction
    mask = bg_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
#           cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
           x, y, w, h = cv2.boundingRect(cnt)
           list.append([x,y,w,h])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x1,y1,x2,y2,id=bbox
        cx=int(x1+x1+x2)//2
        cy=int(y1+y1+y2)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           going_in[id]=(cx,cy)
        if id in going_in:   
           if cy2<(cy+offset) and cy2>(cy-offset):
              cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
              cv2.rectangle(frame, (x1, y1), (x2+x1, y2+y1), (255, 0, 0), 3)
              cvzone.putTextRect(frame,f'{id}',(x1,y1),2,2)
              if counter1.count(id)==0:
                 counter1.append(id)
                 
        if cy2<(cy+offset) and cy2>(cy-offset):
            going_out[id]=(cx,cy)
        if id in going_out:
           if cy1<(cy+offset) and cy1>(cy-offset):
              cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
              cv2.rectangle(frame, (x1, y1), (x2+x1, y2+y1), (255, 0, 255), 3)
              cvzone.putTextRect(frame,f'{id}',(x1,y1),2,2)
              if counter2.count(id)==0:
                 counter2.append(id)  
            
            
    cv2.line(frame,(10,cy1),(1024,cy1),(255,255,255),2)
    cv2.line(frame,(4,cy2),(1018,cy2),(255,255,255),2)
    print(counter1)
    print(counter2)
    cv2.imshow('RGB', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
