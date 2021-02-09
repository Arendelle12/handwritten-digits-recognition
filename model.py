from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt

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

model.save('my_model_2.h5')

# wizualizacja na wykresie
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
