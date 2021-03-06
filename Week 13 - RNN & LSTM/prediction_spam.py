import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle
from keras.models import load_model


from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam-utf8.csv')
# Keeping only the neccessary columns
data = data[['v1','v2']]

data['text'] = data['v2']
data['classification'] = data['v1']
data = data[['text','classification']]


data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# for idx, row in data.iterrows():
#     row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196


labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['classification'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)


model_to_load = "spam_analyzer.h5"
model = load_model(model_to_load)

def make_prediction_on_text(text:str):

    print("Predicting on...")
    print(text)
    text = text.lower()
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    # print("Cleaned...")
    # print(text)
    x = tokenizer.texts_to_sequences([text])
    x = pad_sequences(x, maxlen=152)

    # print("shape", x.shape)
    # print("shape", x.shape[1])
    x = x[[0],:]

    # print("Tokenized...")
    # print(x)
    prediction = model.predict(x)[0]
    print("Prediction...")
    # prediction = model.predict_classes(x)
    # print(prediction)
    for key, value in int_to_spam_class.items():
        print(value, ": ", prediction[key])


int_to_spam_class = {
    0: "ham",
    1: "spam"
}

make_prediction_on_text("Hey are you free for dinner tomorrow night?")
make_prediction_on_text("Want sex with single woman in area now? Click or text the link below!")
make_prediction_on_text("want to make naughty money right now? Click here! Link below! Want some sex please? Text the number below to talk to Hot singles woman near you!")
