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
decoded = Dense(int(encoding_dim /2), activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist, fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# visualize

from matplotlib import pyplot as plt

plt.imshow(x_test[1].reshape(28,28))
plt.title("Image to Be Encoded")
plt.show()


x = x_test[[1],:]

prediction = autoencoder.predict(x)

plt.imshow(prediction.reshape(28,28))
plt.title("Image Decoded")
plt.show()

model_name = "autoencoder"
print("Rendering Loss/Acc Trends...")
for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    plt.show()
