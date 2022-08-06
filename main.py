#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[53]:


# Pre-Declaration :

PREVIEW     = 0
BLUR        = 1
FACE_DETECT = 2
CANNY       = 3

image_filter = PREVIEW
result = None
alive = True

# Creating Window :

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

source = cv2.VideoCapture(0)

# Mode 1 : Face Detection 

def face_detect(frame):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                   "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    # Model parameters :
    
    in_width = 300
    in_height = 300
    mean = [104, 117, 123]
    conf_threshold = 0.7

    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame :
        
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)
        
    # Run a model :
        
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence : %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                    (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                    (255, 255, 255), cv2.FILLED)
          
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        t, _ = net.getPerfProfile() 
        label = 'Inference time : %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (180, 450), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0))
        
        label = 'FACE DETECTION'
        cv2.putText(frame, label, (180, 35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))

        
        return frame

# Mode 2 : Blur 

def blur_filter(frame):
    frame = cv2.blur(frame, (13,13))
    label = 'BLUR MODE'
    cv2.putText(frame, label, (230, 35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
    
    return frame


# Mode 3 : Canny (Edge Detector)

def canny_filter(frame):
    frame = cv2.Canny(frame, 80, 90)
    label = 'CANNY MODE'
    cv2.putText(frame, label, (220, 35), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
    
    return frame

# Filter Selection :

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = canny_filter(frame)
    elif image_filter == BLUR:
        result = blur_filter(frame)
    elif image_filter == FACE_DETECT:
        result = face_detect(frame)
        
    # Display frame :    
                
    cv2.imshow(win_name, result)
    
    # Key Interrupts :
    
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FACE_DETECT
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
        
source.release()
cv2.destroyWindow(win_name)

