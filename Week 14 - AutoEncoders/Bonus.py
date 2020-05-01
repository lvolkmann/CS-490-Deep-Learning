from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# make deep
encoded = Dense(int(encoding_dim /2), activation='relu')(encoded)
encoded = Dense(int(encoding_dim /4), activation='relu')(encoded) # middle layer

# this model maps an input to its reconstruction
encoder = Model(input_img, encoded)

from keras.datasets import mnist, fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# visualize

model_name = "encoder"

from matplotlib import pyplot as plt

plt.imshow(x_test[1].reshape(28,28))
plt.title("Image to Be Encoded")
file_name = model_name + "_to_be_encoded.png"
plt.savefig("output\\" + file_name)
plt.show()


x = x_test[[1],:]

prediction = encoder.predict(x)

plt.imshow(prediction)
plt.title("Image Encoded")
file_name = model_name + "_encoded.png"
plt.savefig("output\\" + file_name)
plt.show()
