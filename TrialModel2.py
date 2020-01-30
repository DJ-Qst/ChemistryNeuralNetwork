import numpy as np
from tensorflow import keras

xdata = "features12.npy"
ydata = "labels12.npy"
numLoops = 250
accuracies = []

X = np.load(xdata)
y = np.load(ydata)
X = X / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50)),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(64, activation="softplus"),

    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
for i in range(3):
    model.fit(X, y, epochs=numLoops)
    accuracies.append(model.evaluate(X,y))

print(accuracies)