from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os


def sliding_window(image,step_size,window_size):
    for y in range(0,image.shape[0],step_size):
        for x in range(0,image.shape[1],step_size):

            yield(x,y,image[y:y+window_size[1],x:x+window_size[0]])



model = joblib.load('helmet_detector.npy')

scale = 0
detections = []

filename = 'Helmet-Reflection-Man-2780x2780.jpg'
img = cv2.imread(filename,0)

window_size = (128,128)
down_scale = 1.5

stepSize = 5
img= cv2.resize(img,(400,400))
for resized in pyramid_gaussian(img,downscale=down_scale):
    for(x,y,window) in sliding_window(resized,step_size=stepSize,window_size=window_size):
        if window.shape[0] != window_size[0] or window.shape[1] !=window_size[1]: 
            continue
        hog_img =  hog(window,orientations = 10,pixels_per_cell = (7, 7),cells_per_block = (3, 3),block_norm='L1',transform_sqrt=True)
        
        pred = model.predict([hog_img])

        hog_img = hog_img.reshape(1,-1)
        if pred==1:
            if model.decision_function(hog_img) > 0.5:
                print(f"Confidence score:{scale, model.decision_function(hog_img)*100}%")
                detections.append((int(x * (down_scale**scale)), int(y * (down_scale**scale)), model.decision_function(hog_img),int(window_size[0]*(down_scale**scale)),int(window_size[1]*(down_scale**scale))))

    scale += 1

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  

sc = [score[0] for (x, y, score, w, h) in detections]
sc = np.array(sc)

pick = non_max_suppression(rects, probs = sc, overlapThresh = -0.5)

for (x1, y1, x2, y2) in pick:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow('my_model',img)
cv2.imwrite('example\\'+filename,img)
k = cv2.waitKey(0) & 0xFF 

if k == 27:             
    cv2.destroyAllWindows()
