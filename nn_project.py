# Create your first MLP in Keras
import os

import numpy
from keras.layers import Dense
from keras.models import Sequential

# fix random seed for reproducibility
numpy.random.seed(7)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/dataset2.txt')

dataset = numpy.loadtxt(filename, delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:3]
Y = dataset[:, 3]
# create model
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=50, batch_size=1)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
