# -*- coding: utf-8 -*-
"""housing-price-using-neural-network.ipynb

# Predict the housing price for Boston housing price problem
### Step1: Import all the required libraries
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

"""### Step2: Load dataset
#### The dataset is, in fact, not in CSV format in the UCI Machine Learning Repository. The attributes are instead separated by whitespace. We can load this easily using the pandas library. Then split the input (X) and output (Y) attributes, making them easier to model with Keras and scikit-learn.
"""

dataframe = pd.read_csv("https://raw.githubusercontent.com/daudayal1987/aiandml/refs/heads/main/housing_boston.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]

#Create train, validation and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

X_train.shape

X_test.shape

X_valid.shape

y_train.shape

y_test.shape

y_valid.shape

"""### Step3: Define model


"""

#define layers
hidden_layer1 = Dense(13, input_shape=(13,), kernel_initializer="normal", activation="relu")
output_layer = Dense(1, kernel_initializer="normal")

#define model
model = Sequential()

#add layers to model
model.add(hidden_layer1)
model.add(output_layer)

#compile model
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error'])

"""### Step4: Print metadata for model"""

model.summary()

for i, layer in enumerate(model.layers):
    if hasattr(layer, 'kernel_initializer'):
        print(f"Layer {i} - {layer.name}:")
        print("  Kernel Initializer:", layer.kernel_initializer)
        print()

print("\n📌 Initial Weights:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer {layer.name} weights:\n", weights[0])
        print(f"Layer {layer.name} biases:\n", weights[1])

"""### Step5: Train the model"""

history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid))

"""### Step 6: Evaluate the model"""

pd.DataFrame(history.history)[['loss', 'val_loss', 'root_mean_squared_error', 'val_root_mean_squared_error']].plot(figsize=(8,5))
plt.grid(True)
plt.show()

pd.DataFrame(history.history)[['mean_absolute_percentage_error', 'val_mean_absolute_percentage_error']].plot(figsize=(8,5))
plt.grid(True)
plt.show()

#evaluate the model
scores = model.evaluate(X_test, y_test)
print("metrices", model.metrics_names)
print(scores)
# print("\n%s: %.2f%%" % (model.metrics_names[0], scores))

for layer in model.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    if weights:
        print(f"  Weights shape: {weights[0].shape}")
        print(f"  Weights: \n{weights[0]}")
        print(f"  Biases shape: {weights[1].shape}")
        print(f"  Biases: \n{weights[1]}")
    else:
        print("  No weights (e.g., Dropout or Flatten layer)")

all_weights = model.get_weights()
for i, param in enumerate(all_weights):
    print(f"Param {i} shape: {param.shape}\n{param}\n")