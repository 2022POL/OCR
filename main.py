from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

from matplotlib import pyplot as plt
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from matplotlib import cm

# 이미지 파일 경로
# image_path = os.getenv('')

# 이미지 흑백으로 열기 (pillow)
img = Image.open("a-1.PNG").convert('L')

# 이미지를 784개 픽셀로 사이즈 변환
img = np.resize(img, (1, 784))

test_data = ((np.array(img)/255)-1)*(-1)
# test_data = np.array(img)

for x in img:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)


model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))



res = (model.predict(test_data) > 0.5).astype("int32")
# r = model.predict(test_data).astype("float64")
# res = r.argmax(axis=-1)

print(res)


"""
y_vloss = history.history['val_loss']

y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
"""

"""
print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))

plt.imshow(X_train[0], cmap='Greys')
plt.show()

for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')


print("class : %d" % (Y_class_train[0]))



print(Y_train[0])
"""