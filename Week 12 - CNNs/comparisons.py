from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import numpy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K


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

model_to_load = "original.h5"
model = load_model(model_to_load)
loss, accuracy = model.evaluate(X_test, y_test)


model_to_load = "new.h5"
model = load_model(model_to_load)
loss2, accuracy2 = model.evaluate(X_test, y_test)


print("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL ACCURACY: {accuracy}
CHANGE: {accuracy_change}
NEW ACCURACY: {accuracy2}
""".format(loss=loss, loss2=loss2, loss_change=loss2-loss, accuracy=accuracy, accuracy_change= accuracy2 - accuracy, accuracy2=accuracy2))

input("Continue?...")

history_to_load = "new_history.p"
history = pickle.load( open(history_to_load, "rb"))

print("Rendering Loss/Acc Trends...")
for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    plt.show()