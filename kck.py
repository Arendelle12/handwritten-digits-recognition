from keras import models
import numpy as np
import cv2
from skimage.color import rgb2gray

model = models.load_model('my_model_2.h5')

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
        if drawing:
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

    # quit
    if key == ord('q'):
        break
    # clear window
    elif key == ord('c'):
        img = np.zeros((512, 512, 3), np.uint8)

    # predict number
    elif key == ord('p'):
        gray = rgb2gray(img)
        image = cv2.resize(gray, (28, 28))
        final = np.reshape(image, (1, 28, 28, 1)).astype('float32')
        predictions = model.predict([final])
        # predictions
        print("PREDICTION : ", np.argmax(predictions))

cv2.destroyAllWindows()
