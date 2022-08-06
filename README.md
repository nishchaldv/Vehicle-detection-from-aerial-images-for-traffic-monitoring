Deep learning approaches have demonstrated state-of-the-art performance in various computer vision tasks such as object detection and recognition.
The Convolutional Neural Network (CNN) used. The CNN accepts as input an image patch of 50x50 pixels. The patches are extracted using the sliding window appoach.
To identify potential regions of interest such as the road, which is more probable to containg vehicles color thresholding is performed. But first it is necessary to identify the color regions that represent the area we are looking for. For this reason the sliders_color.py implements a GUI which takes as input an image and uses slider bars for the minimum and maximum pixels values per 3 color channesl in order to identigy the range of colors to isolate. The specific chromatic model used is the HSV model.
The dataset used is a subset of a larger dataset collected using a DJI Matrice 100 UAV which is obtained from kaggle. The vehicle images where cropped and used to construct a training and validation set.
Dependencies
- Python - 3.6.4
- Keras - 2.2.0
- Tensorflow - 1.5.0
- Numpy - 1.14.5
- OpenCV - 3.4.0
The color thresholding is performed on the provided image. The best values will need to be passed through to the detection stage. The detection stage has two modes one using the mask and one with just the sliding window. The window size, stride of the window. 
