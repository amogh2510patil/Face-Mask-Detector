# import the necessary packages
import numpy as np
import time
import cv2



#defining prototext and caffemodel paths
caffeModel = "D:/amogh/python_projects/ML/Face_Detection/Face-detection-with-OpenCV-and-deep-learning-master/models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "D:/amogh/python_projects/ML/Face_Detection/Face-detection-with-OpenCV-and-deep-learning-master/models/deploy.prototxt.txt"

#Load Model
print("Loading model...................")
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

# initialize the video stream to get the video frames
print("[INFO] starting video stream...")
camera=cv2.VideoCapture(0)
time.sleep(2.0)


#loop the frams from the  VideoStream
while True :
    
    _,image=camera.read()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # Identify each face
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, save it as a separate file
        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]
        # show the output frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
camera.release()
cv2.destroyAllWindows()
