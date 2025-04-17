#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences 

#load data
path= "/home/pkar/regres/data_preprocesing_trim.csv"
df_bind = pd.read_csv(path)

df_bind = df_bind.sample(frac=1) #shuffle a DataFrame rows

lines = df_bind['to_a'].values.tolist()  #connected SMILE and aminoacid sequece
review_lines = list()
for line in lines:
    review_lines.append(line)
    
MAX_SEQUENCE_LENGTH = 800

tokenizer = Tokenizer(lower = False, char_level=True)
tokenizer.fit_on_texts(review_lines)
sequences = tokenizer.texts_to_sequences(review_lines)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating = 'post')

sentiment =  df_bind['logP'].values
print('Shape of data tensor:', review_pad.shape)
print('Shape of label tensor:', sentiment.shape)

training_samples = round(len(review_lines)*2/3)
validation_samples = round(len(review_lines)/3)
print ('liczba trenowania', training_samples)

x_train = review_pad[:training_samples]
y_train = sentiment[:training_samples]
x_test = review_pad[training_samples: training_samples + validation_samples]
y_test = sentiment[training_samples: training_samples + validation_samples]

from keras import layers
from keras import models
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, BatchNormalization
from keras.initializers import Constant
#from tensorflow.keras.layers import Dense, BatchNormalization

model2 = Sequential()
model2.add(Embedding(10000, 8, input_length=800))
model2.add(Conv1D(filters=64, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=64, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=64, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))


model2.add(Conv1D(filters=64, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=32, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=32, kernel_size=8, activation=layers.LeakyReLU(alpha=0.01)))
model2.add(BatchNormalization())
model2.add(MaxPooling1D(pool_size=2))

model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1))
print(model2.summary())


from keras.optimizers import RMSprop
model2.compile(optimizer=RMSprop(), loss='mae')
history = model2.fit(x_train, y_train,
epochs=200,
validation_data=(x_test, y_test))

model2.save('/home/pkar/regres/final_modellog_mae.h5')


import matplotlib.pyplot as plt
acc = history.history['loss']
val_acc = history.history['val_loss']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Loss train')
plt.plot(epochs, val_loss, 'b', label='Loss validation')
plt.title('Loss train and validation')
plt.xlabel('Epoch')
plt.ylabel('Loss -log(Ki)')
plt.legend()
plt.savefig('/home/pkar/regres/final_modellog_mae.jpeg')

