# imports
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
from google.colab.patches import cv2_imshow
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import preprocess_input

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model('/content/drive/My Drive/bestsnakev1.model')
# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images('/content/drive/My Drive/splitdata/testing'))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]
# initialize list of results
results = []
# loop over sampled image paths
for p in imagePaths:
	# load original input image
	orig = cv2.imread(p)
	# pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]

	# order channel dimensions (channels-first or channels-last)
	# depending on the Keras backend, then add a batch dimension to
	# the image
	image = load_img(p, target_size=(150, 150))
	image = img_to_array(image)  #output Numpy-array

	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

	image = preprocess_input(image)
	x = model.predict(image)[0]
	
	g = ("{:.1f}%".format(x[0] * 100))
	
	
	
	label = "snake" if x[0] > .85 else "no snake"
	color = (0, 0, 255) if x[0] < .85 else (0, 255, 0)
  
	# resize original input (to better visualize it) and
	# then draw the label on the image
	orig = cv2.resize(orig, (128, 128))
	cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.putText(orig, g, (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	# add the output image to our list of results
	results.append(orig)


# create a montage using 128x128 "tiles" with 4 rows and 4 columns
montage = build_montages(results, (128, 128), (4, 4))[0]
# show the output montage
cv2_imshow(montage)
cv2.waitKey(0)
