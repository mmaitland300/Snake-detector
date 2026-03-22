# Legacy script archived from early Colab prototype.

import random

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from imutils import build_montages, paths
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

print("[INFO] loading pre-trained network...")
model = load_model("/content/drive/My Drive/bestsnakev1.model")
imagePaths = list(paths.list_images("/content/drive/My Drive/splitdata/testing"))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]
results = []
for p in imagePaths:
    orig = cv2.imread(p)
    image = load_img(p, target_size=(150, 150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    x = model.predict(image)[0]

    g = "{:.1f}%".format(x[0] * 100)
    label = "snake" if x[0] > 0.85 else "no snake"
    color = (0, 0, 255) if x[0] < 0.85 else (0, 255, 0)

    orig = cv2.resize(orig, (128, 128))
    cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(orig, g, (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    results.append(orig)

montage = build_montages(results, (128, 128), (4, 4))[0]
cv2_imshow(montage)
cv2.waitKey(0)
