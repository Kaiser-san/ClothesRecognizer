import cv2
import keras
import numpy as np
import sys

from skimage import measure

from keras.preprocessing.image import img_to_array, load_img

# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
img = cv2.imread('tests/{}.png'.format(tp_idx))

#################################################################################
# U ovoj sekciji implementirati obradu slike, ucitati prethodno trenirani Keras
# model, i dodati bounding box-ove i imena klasa na sliku.
# Ne menjati fajl van ove sekcije.

# Ucitavamo model
model = keras.models.load_model('model.h5')

print(cv2.__version__)


classes = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

kernel = np.ones((2, 2), np.uint8)

openedOnce = cv2.morphologyEx(cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.MORPH_OPEN, kernel,
                              iterations=1)
thresh = cv2.threshold(openedOnce, 10, 255, cv2.THRESH_BINARY)[1]
openedOnce = cv2.bitwise_not(openedOnce)
openedTwice = cv2.morphologyEx(openedOnce, cv2.MORPH_OPEN, kernel, iterations=1)

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
shapeCoords = []

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue

    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask);
        labelMask = cv2.cvtColor(labelMask, cv2.COLOR_GRAY2BGR)
        white_pixels = np.where(labelMask == 255)
        coords = np.column_stack((white_pixels[1], white_pixels[0]))
        rect = cv2.minAreaRect(coords)
        box = np.int0(np.around(cv2.boxPoints(rect)))
        shapeCoords.append(box)


def process_img(crop_img, w, h):
    if (w > h):
        w1 = 27
        w1_pos = 0
        r = w / w1
        h1 = h / r
        h1_pos = (27 - h1) / 2
    else:
        h1 = 27
        h1_pos = 0
        r = h / h1
        w1 = w / r
        w1_pos = (27 - w1) / 2
    w1 = int(w1)
    h1 = int(h1)
    w1_pos = int(w1_pos)
    h1_pos = int(h1_pos)
    crop_img = cv2.resize(crop_img, (w1, h1))
    blank_image = 255 * np.ones((28, 28), np.uint8)
    blank_image[h1_pos:h1 + h1_pos, w1_pos:w1 + w1_pos] = crop_img

    return blank_image


import math

for shapeCoords in shapeCoords:
    x = shapeCoords[0][0]
    y = shapeCoords[1][1]
    w = abs(shapeCoords[2][0] - x)
    h = abs(shapeCoords[3][1] - y)

    crop_img = openedTwice[y:y + h, x:x + w]
    crop_img = process_img(crop_img, w, h)

    crop_img = img_to_array(crop_img)
    crop_img = crop_img[np.newaxis, ...]

    crop_img = crop_img.astype('float32')
    crop_img /= 255

    clss = model.predict(crop_img)
    cls_val = classes[np.argmax(clss[0])]

    cv2.drawContours(img, [shapeCoords], 0, 255, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 0.33
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(img, cls_val,
                (shapeCoords[1][0], math.floor((shapeCoords[0][1] - shapeCoords[1][1]) / 2) + shapeCoords[1][1]),
                font,
                fontScale,
                fontColor,
                lineType)

solution = img.copy()

#################################################################################

# Cuvamo resenje u izlazni fajl
cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)