# importy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray
"""
# zbiór treningowy i zbiór testowy
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# przekształcenie x (1 - skala szarości)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# przekształcenie y - kodowanie cyfr (1 tam, gdzie jest cyfra)
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# model sieci konwolucyjnej
model = Sequential()
# wastwy sieci
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# trenowanie sieci
# 3 iteracje po całym zbiorze
# 32 próbki treningowe na gradient
# fit == train
hist = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=3)

model.save('my_model_2.h5')"""

model = models.load_model('my_model_2.h5')

# wizualizacja na wykresie
"""plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()"""

# #############################
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
drawing = False  # true if mouse is pressed
ix, iy = -1, -1


# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing  # , mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:

            cv2.circle(img, (x, y), 17, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while 1:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        img = np.zeros((512, 512, 3), np.uint8)
        
    elif key == ord('p'):
        # print(img.shape)
        gray = rgb2gray(img)
        # print(gray.shape)
        image = cv2.resize(gray, (28, 28))
        # print(image.shape)
        # cv2.imshow('test', image)
        final = np.reshape(image, (1, 28, 28, 1)).astype('float32')
        #plt.imshow(final)
        predictions = model.predict([final])
        # predictions
        print("PREDICTION : ", np.argmax(predictions))




cv2.destroyAllWindows()
