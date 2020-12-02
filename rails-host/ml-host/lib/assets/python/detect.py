import sys
import cv2
import numpy as np
from tensorflow.keras import models
input = sys.argv[1]
img = cv2.imread(input)
input_image = np.expand_dims(img, axis= 0)

model = models.load_model('/home/christian/Documents/ML-Final-Project/rails-host/ml-host/lib/assets/python/new_new.h5')

prediction = int(model.predict_classes(input_image))
if prediction == 0:
    print ("Chair")
else:
    print("Bed")


# cv2.imshow("Test", img)
# cv2.waitKey(0)
