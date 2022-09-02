# %% [markdown]
# # NLP Classification Algorithm

# %%
from IPython.display import display
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.layers.merge import add
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 6)


# %% [markdown]
# # Prepare data

# %%
# load data

df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.head()

# %%
cates = df.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())

# %%
# as shown above, THE WORLDPOST and WORLDPOST should be the same category, so merge them.

df.category = df.category.map(
    lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

# %%
# using headlines and short_description as input X

df['text'] = df.headline + " " + df.short_description

# tokenizing

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data

df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()

# %%
df.word_length.describe()

# %%
# using 50 for padding length

maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))

# %%
# category to id

categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k: i})
    int_category.update({i: k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])

# %% [markdown]
# # GloVe Embedding

# %%
word_index = tokenizer.word_index

EMBEDDING_DIM = 100

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))

# %%
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)

# %% [markdown]
# # Dataset Split
#

# %%
# prepared data

X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

# and split to training set and validation set

seed = 29
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=seed)

# %% [markdown]
# # LSTM with Attention

# %%
# from https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043/code


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


lstm_layer = LSTM(300, dropout=0.25, recurrent_dropout=0.25,
                  return_sequences=True)

inp = Input(shape=(maxlen,), dtype='int32')
embedding = embedding_layer(inp)
x = lstm_layer(embedding)
x = Dropout(0.25)(x)
merged = Attention(maxlen)(x)
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.25)(merged)
merged = BatchNormalization()(merged)
outp = Dense(len(int_category), activation='softmax')(merged)

AttentionLSTM = Model(inputs=inp, outputs=outp)
AttentionLSTM.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['acc'])

AttentionLSTM.summary()

# %%
attlstm_history = AttentionLSTM.fit(x_train,
                                    y_train,
                                    batch_size=128,
                                    epochs=20,
                                    validation_data=(x_val, y_val))

# %%
acc = attlstm_history.history['acc']
val_acc = attlstm_history.history['val_acc']
loss = attlstm_history.history['loss']
val_loss = attlstm_history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()

plt.show()

# %%
# confusion matrix

predicted = AttentionLSTM.predict(x_val)
cm = pd.DataFrame(confusion_matrix(
    y_val.argmax(axis=1), predicted.argmax(axis=1)))

# %%
pd.options.display.max_columns = None
display(cm)

# %%
d = {'sentence': ["Hold a public inquiry into Government contracts granted during Covid-19",
                  "Force universities to offer a full refund to university students du to Covid-19 pandemic", "Make LGBT conversion therapy illegal in the UK"]}
test_df = pd.DataFrame(d)
test_df

# %%
# test_sentence = "Hold a public inquiry into Government contracts granted during Covid-19"
# tokenizer.fit_on_texts(df.text)
X_test = tokenizer.texts_to_sequences(test_df['sentence'])
foo = list(sequence.pad_sequences(X_test, maxlen=50))
foo = np.array(foo)

# %%
predic = AttentionLSTM.predict(foo)
y_predic = predic.argmax(axis=-1)

# %%
for i in range(len(y_predic)):
    print(categories[y_predic[i]] + " : " + test_df['sentence'][i])

# %% [markdown]
# ## Evaluate Accuracy

# %%


def evaluate_accuracy(model):
    predicted = model.predict(x_val)
    diff = y_val.argmax(axis=-1) - predicted.argmax(axis=-1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects/total)


# %%
print("model LSTM with Attention:       %.6f" %
      evaluate_accuracy(AttentionLSTM))
