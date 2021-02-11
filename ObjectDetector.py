import numpy as np
import cv2
import tensorflow as tf
import serial
from scipy.spatial import distance
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import time


def colorConvert(image):
    return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# Initialize capture
cap = cv2.VideoCapture(0)
# Define and set frame dimensions
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Define a boolean for background capture
firstRun = True
# Define some variables
kernel = np.array((5, 5))  # filter for dilation
# Blur thresholds to mitigate the effects of motion on object detection
minBlur = 350
maxBlur = 3000
# Define cosine similarity threshold
cosThresh = 0.85
# Initialize some lists
obj_emb_mem = []
for_PCA = []
labels = []
# Set to true to enable track
track = False

if track:   # initializes serial communication with arduino
    arduino = serial.Serial('COM4', 9600)   # create serial object named arduino

# Load Model
net = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

while True:
    validCont = []
    if firstRun:
        frames = []
        print("Capturing background, please keep stable")
        for count in range(15):
            success, frame = cap.read()
            frames.append(frame)
        bg = np.average(frames, axis=0).astype(dtype=np.uint8)
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        firstRun = False
        print("Background captured")
    else:

        success, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bg_removed_frame = cv2.absdiff(gray_frame, bg_gray)

        frame_blur = cv2.GaussianBlur(bg_removed_frame, (11, 11), 0)

        ret, frame_threshold = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        frame_threshold = cv2.dilate(frame_threshold, kernel, iterations=2)

        blur = cv2.Laplacian(frame_threshold, cv2.CV_64F).var()

        (contours, _) = cv2.findContours(frame_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            area = cv2.contourArea(i)
            if area > 4000 and blur > minBlur and blur < maxBlur:

                x, y, width, height = cv2.boundingRect(i)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (123, 0, 255), 2)

                # Get features
                temp_crop = frame[y:(y + height), x:(x + width)]
                cv2.imshow('Crop Out', temp_crop)
                embeds = net(temp_crop)
                embeds = embeds.numpy()
                # save feature for subsequent PCA analysis
                for_PCA.append(embeds)

                if len(obj_emb_mem) == 0:
                    obj_emb_mem.append(embeds)
                    labels.append(int(0))
                else:
                    temp_dist = []
                    for idx, obj in enumerate(obj_emb_mem):
                        temp_dist.append(distance.cosine(obj, embeds))

                    dis = min(temp_dist)
                    if dis > cosThresh:
                        cv2.putText(frame, 'NOT SAME OBJECT', (x, y+15), 1, 1, (255, 0, 0), 1)
                        obj_emb_mem.append(embeds)
                        print(len(obj_emb_mem))
                    else:
                        dis_min_idx = temp_dist.index(dis) + 1
                        cv2.putText(frame, 'SAME OBJECT ID ' + str(dis_min_idx), (x, y+15), 1, 1, (255, 0, 0), 1)

                    # save label for subsequent PCA analysis
                    labels.append(temp_dist.index(dis) + 1)

        if len(validCont) == 1 and track:
            x, y, width, height = cv2.boundingRect(validCont[0])
            command = frameWidth / 2 - int(x + width / 2)
            print(command)
            arduino.write(str.encode(str(command)))
            time.sleep(1.5)

        cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Get a numpy array of feature vectors
np_obj_arr = np.reshape(np.array([obj for obj in for_PCA]), (len(for_PCA), 1280))

# define pca decomposition
pca = decomposition.PCA()
pca.n_components = 2
# find the first two principal components of the data for plotting
data = StandardScaler().fit_transform(np_obj_arr)
pca_data = pca.fit_transform(data)
# add labels for plotting
pca_data = np.vstack((pca_data.T, labels)).T
# get a data frame
pca_df = pd.DataFrame(data=pca_data, columns=("First Princ.", "Second Princ.", "Label"))
# plot data
sn.FacetGrid(pca_df, hue="Label", height=6).map(plt.scatter, "First Princ.", "Second Princ.").add_legend()
plt.show()
