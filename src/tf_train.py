import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.backend import image_data_format

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="NanumBarunGothic.ttf").get_name()
rc('font', family=font_name)

DATA_DIR = 'data/image_merge'
IMG_SIZE = 64
LR = 0.001

MODEL_NAME = 'trademarks-{}-{}.model'.format(LR, '2conv-basic')
MODEL_NAME = MODEL_NAME + '.h5'


def label_img(img):
    word_label = img.split('.')[-2]
    if word_label == '0':
        return [1, 0]
    elif word_label == '1':
        return [0, 1]
    else:
        pass


def title_img(img):
    title1 = img.split('.')[0]
    title2 = img.split('.')[1]
    return title1, title2


def create_train_test_data():
    all_data = []
    train_data = []
    test_data = []
    for img in tqdm(os.listdir(DATA_DIR)):
        label = label_img(img)
        title1, title2 = title_img(img)
        path = os.path.join(DATA_DIR, img)
        print(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        all_data.append([np.array(img), np.array(label), title1, title2])
    shuffle(all_data)

    try:
        train_data = all_data[:-20]
        shuffle(train_data)
        np.save('train_data.npy', train_data)
        test_data = all_data[-20:]
        shuffle(test_data)
        np.save('test_data.npy', test_data)
    except Exception:
        pass

    return train_data, test_data


train_data, test_data = create_train_test_data()
# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')

if image_data_format() == 'channels_first':
    input_shape = (3, IMG_SIZE, IMG_SIZE)
else:
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

model = Sequential()

# 1st layer
model.add(Conv2D(32, 5, input_shape=input_shape, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# 2nd layer
model.add(Conv2D(64, 5, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# 3rd layer
model.add(Conv2D(32, 5, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# 4th layer
model.add(Conv2D(64, 5, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# 5th layer
model.add(Conv2D(32, 5, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# 6th layer
model.add(Conv2D(64, 5, activation='relu', padding='same'))
model.add(MaxPooling2D((5, 5), padding='same'))

# Fully Connected
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.8))

# Classify whether it is same or not same
model.add(Dense(2))
model.add(Activation('softmax'))

# Use Adam Optimizer
opt = Adam(LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())


train = train_data[:-5]
validation = train_data[-5:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]
X_train = np.array([x for x in X])
y_train = np.array([y for y in Y])

val_x = np.array([i[0] for i in validation]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
val_y = [i[1] for i in validation]
x_val = np.array([x for x in val_x])
y_val = np.array([y for y in val_y])

model.fit(X_train, y_train,
          shuffle=True,
          epochs=30,
          validation_data=(x_val, y_val),
          verbose=2,

          )

model.save_weights(MODEL_NAME)

# Serialize Model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

fig = plt.figure()

for label, data in enumerate(test_data[:10]):

    img_label = data[1]
    img_data = data[0]
    title1 = data[2]
    title2 = data[3]

    y = fig.add_subplot(3, 4, label + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    data = np.expand_dims(data, axis=0)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Non-similar'
    else:
        str_label = 'Similar'

    plt_subtitle = str_label + str(img_label) + "\n" + title1 + ' & ' + title2
    y.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title(plt_subtitle)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
