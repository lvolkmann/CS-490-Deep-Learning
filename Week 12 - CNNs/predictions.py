import numpy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

# Thanks: https://stackoverflow.com/questions/57797113/attributeerror-module-keras-backend-has-no-attribute-image-dim-ordering
K.common.set_image_dim_ordering('th') # no more att error with keras 2.x

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

test_images = X_test
test_labels = y_test

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


num_to_text = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

model_to_load = "original.h5"
model = load_model(model_to_load)



def make_prediction(i):
    print("Rendering test image...")
    test_img = test_images[i]
    test_data= X_test[[i], :]

    plt.imshow(test_img, cmap=plt.get_cmap('gray'))
    plt.title("Model Prediction: {}".format(num_to_text[model.predict_classes(test_data)[0]]))
    plt.show()

prediction_idx = [50, 75, 100, 122]

for idx in prediction_idx:
    make_prediction(idx)