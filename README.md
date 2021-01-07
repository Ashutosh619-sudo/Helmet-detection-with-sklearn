# Helmet-detection-with-sklearn
Using sklearn and hog feature descriptor to detect bike helmet.

I have used Linear Support vector machine for the model.
To enhance the classifier I have used hog(histogram of oriented gradient) https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients, it is a popular technique for object and is used a lot. There are very less examples of object detections with sklearn.

I have used sliding window technique to search for objects through the image and at every iteration used image pyramid to scale it down.

#Requirements:
1.sklearn
2.sklearn-image
3.opencv
4.imutils

Its not perfect but it works.
