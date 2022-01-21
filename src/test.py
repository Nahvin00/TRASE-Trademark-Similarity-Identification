import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from text_to_img_2 import save_pair_image
from romanize import romanize
import re

def proc(inp1,inp2):

    MAX_LENGTH = 20
    LR = 0.001
    IMG_SIZE = 64  # 224
    MODEL_NAME = 'trademarks-{}-{}.model'.format(LR, '2conv-basic')
    MODEL_NAME = MODEL_NAME + '.h5'

    title1_roman = romanize(inp1)
    title1_input = title1_roman

    title2_roman = romanize(inp2)
    title2_input = title2_roman

    save_pair_image(title1_input, title2_input, 0)
    title1_refine = re.sub(r"[`;&<>~0-9\[\]|:+-/.,!?@#$%^*()\'\"]", r"", title1_input)
    title2_refine = re.sub(r"[`;&<>~0-9\[\]|:+-/.,!?@#$%^*()\'\"]", r"", title2_input)

    path = 'D:\\PythonProject\\phonetic\\image_test\\' + str(title1_refine[:MAX_LENGTH]) + "." + str(
        title2_refine[:MAX_LENGTH]) + '.' + str(0) + '.png'

    data = []
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img_fourD = np.expand_dims(img, axis=0)
    data.append(img_fourD)

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    if os.path.exists(MODEL_NAME):
        loaded_model.load_weights(MODEL_NAME)
        print("Loaded model from disk")

    y_prob = loaded_model.predict(data)
    model_out_classes = y_prob.argmax(axis=-1)

    if model_out_classes == 1:
        str_label = 'Non-similar'
        confidence = y_prob[0][1]
    else:
        str_label = 'Similar'
        confidence = y_prob[0][0]

    confidence = round(float(confidence), 7)

    print("Prediction: {}\nConfidence: {}".format(str_label, confidence))

    return str_label, confidence


