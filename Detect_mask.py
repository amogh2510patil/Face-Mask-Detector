
import cv2
import torch.nn as nn
import numpy as np
import torch

from torchvision import transforms

dic={0:"MASK",1:"NO MASK"}
colour_dic={0:(0,255,0),1:(0,0,255)}

FILE='model_pth'

model=nn.Sequential(nn.Conv2d(3,16,3),nn.ReLU(),nn.MaxPool2d(2,2),nn.Conv2d(16,32,3),nn.ReLU(),nn.MaxPool2d(2,2),nn.Conv2d(32,16,3),nn.ReLU(),nn.MaxPool2d(2,2),nn.Flatten(),nn.Linear(16*26*26,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(p=0.5),nn.Linear(256,2))
model.load_state_dict(torch.load(FILE,map_location=torch.device('cpu')))
model.eval()

#defining prototext and caffemodel paths
caffeModel = "D:/amogh/python_projects/ML/Face_Detection/Face-detection-with-OpenCV-and-deep-learning-master/models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "D:/amogh/python_projects/ML/Face_Detection/Face-detection-with-OpenCV-and-deep-learning-master/models/deploy.prototxt.txt"

#Load Model
print("Loading model...................")
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

print("[INFO] starting video stream...")
camera=cv2.VideoCapture(0)

while True :
    
    _,image=camera.read()
    image=cv2.flip(image,1)
    
    (h, w) = image.shape[:2]
    try:
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
                frame = image[startY-50:endY+50, startX-50:endX+50]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # show the output frame
                test_img = np.moveaxis(frame,2,0)
            
                transform = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            ])
                test_img=transform(torch.FloatTensor(test_img))/255
                test_imgs = torch.utils.data.DataLoader([test_img], batch_size=1, shuffle=True,num_workers=0)
                
               
                with torch.no_grad():
                    img=next(iter(test_imgs))
                    # for imgs in test_imgs:
                        # pass
                        # imgs = imgs.to(device)
                    output=model(img)
                    _,predicted = torch.max(output.data,1)
                    # image=imgs[0]
                    # plt.imshow(transforms.ToPILImage()(image))
                    # plt.show()
                    # print(predicted)
                
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(image, (startX, startY), (endX, endY),
                            colour_dic[predicted.item()], 2)
                cv2.putText(image, dic[predicted.item()]+'    '+str(text), (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_dic[predicted.item()], 2)
    except:pass
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
camera.release()
cv2.destroyAllWindows()