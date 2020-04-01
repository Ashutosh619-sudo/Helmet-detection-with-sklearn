import numpy as np
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn.externals import joblib
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split
import imutils
import cv2
import os

path_helmet = 'C:\\Users\\abhie\\Desktop\\helmet detector\\data\\helmet'
path_nonhelmet = 'C:\\Users\\abhie\\Desktop\\helmet detector\\data\\non-helmet'

data=[]
labels =[]

for img_file in os.listdir(path_helmet):
    img = cv2.imread(path_helmet+'\\'+img_file,0)
    img = cv2.resize(img,(128,128))
    hog_img = hog(img,orientations = 10,pixels_per_cell = (7, 7),cells_per_block = (3, 3),block_norm='L1',transform_sqrt=True)
    data.append(hog_img)
    labels.append(1)

for img_file in os.listdir(path_nonhelmet):
    img = cv2.imread(path_nonhelmet+'\\'+img_file,0)
    print(img_file)
    img = cv2.resize(img,(128,128))
    hog_img = hog(img,orientations = 10,pixels_per_cell = (7, 7),cells_per_block = (3, 3),block_norm='L1',transform_sqrt=True)
    data.append(hog_img)
    labels.append(0)

shuffle(np.array(data),np.array(labels),random_state=0)

(train_X, test_X, train_y, test_y) = train_test_split(np.array(data),np.array(labels), test_size=0.20, random_state=42)

model = LinearSVC()
print(train_X[0].shape)
model.fit(train_X,train_y)


predictions = model.predict(test_X)

print(classification_report(test_y,predictions))
print(confusion_matrix(test_y,predictions))

joblib.dump(model, 'helmet_detector.npy')

