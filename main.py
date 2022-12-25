from sort import *
import cv2
import numpy as np
import torch 
import tensorflow as tf
from scipy.spatial import distance as dist



detectionModel = torch.hub.load("G:/tensorGO/models/yolov7", 'custom', "G:/tensorGO/models/yolov7_size640_epochs15_batch16/weights/best.pt", source='local', force_reload=True)
maskModel = tf.keras.models.load_model("./models/mask/maskClf.h5", compile=False)
maskModel.trainable = False

MIN_DISTANCE = 50

vidPath = "./samples/sample0.mp4" 
vid = cv2.VideoCapture(vidPath)
colours = np.random.randint(low = 0, high = 255, size = (32, 3))
mot_tracker = Sort()

i = 0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

def predictMasks(df, image):
    preds = []

    for faceCoords in df:
        try:
            face = image[int(faceCoords[0]):int(faceCoords[2]), int(faceCoords[1]):int(faceCoords[3]), :]
            face = cv2.resize(face, (224,224))
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face = np.reshape(face, (-1, 224, 224, 3))
            pred = maskModel.predict(face)
            preds.append(np.argmax(pred))
        except:
            preds.append("")
    return preds



while True:
    ret, image_show = vid.read()
    if(ret):
        preds = detectionModel(image_show)
        detections = preds.pandas().xyxy[0]
        bodyDetections = detections[detections["name"]=="body"].drop(["name"], axis=1).values
        headDetections = detections[detections["name"]=="head"].drop(["name"], axis=1).values
        w = image_show.shape[0]
        h = image_show.shape[1]
        n = len(bodyDetections)
        
        cv2.putText(image_show, f"No of person : {n}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
        

        if n>0:

            maskPreds = predictMasks(headDetections, image_show)

            track_bbs_ids = mot_tracker.update(bodyDetections)
            violate = set()

            centroids = np.array([[(r[0]+r[2])//2, (r[1]+r[3])//2] for r in track_bbs_ids])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)

            for j in range(len(track_bbs_ids.tolist())):
                coords = track_bbs_ids.tolist()[j]
                x1, y1, x2, y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
                name_idx = int(coords[4])
                name = f"ID : {name_idx}"
                color = colours[name_idx%32].tolist()
                thickness = 1
                if j in violate:
                    cv2.putText(image_show, f"Maintain Social Distancing", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)
                    thickness = 3
                    color = (0,0,255)
                cv2.circle(image_show, ((x1+x2)//2, y1), 3, color, 6)
                cv2.putText(image_show, name, ((x1+x2)//2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

            for j in range(len(headDetections)):
                coords = headDetections[j]
                x1, y1, x2, y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
                mask = maskPreds[j]
                thickness = 1
                if mask==0:
                    name = "Mask"
                    color = (0,255,0)
                if mask==1:
                    name = "No Mask"
                    color = (27,27,255)

                cv2.rectangle(image_show, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(image_show, name, ((x1+x2)//2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        cv2.imshow("Video", image_show)
        #image_show = cv2.resize(image_show, (640,480))
        #out.write(image_show)

        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
#out.release()
cv2.destroyAllWindows() 