# Deep-Learning-EX-8

## Implement a Simple LSTM using TensorFlow/Keras

## AIM:

To implement a Long Short-Term Memory (LSTM) neural network using TensorFlow–Keras for sequence prediction, and to understand how LSTM handles long-term dependencies better than traditional RNN models.

## ALGORITHM:

**STEP 1:** Import the required libraries such as NumPy, TensorFlow, Keras Sequential model, LSTM layer, and Dense layer.

**STEP 2:** Create a simple numerical dataset and prepare input–output pairs. Reshape the data into 3D format (samples, timesteps, features) so it can be accepted by the LSTM layer.

**STEP 3:** Build the LSTM neural network using one LSTM layer followed by a Dense layer. Ensure the input shape matches the prepared training data.

**STEP 4:** Compile the model using an appropriate optimizer (Adam) and loss function (MSE). Train the model on the sequence dataset for a fixed number of epochs.

**STEP 5:** Use the trained model to manually validate the output by giving a test sequence and predicting the next value.

**STEP 6:** Plot additional graphs such as the training loss curve and sequence-prediction graph to visually validate the model performance.

## PROGRAM:

### Name: Sanjana Sri N 
### Register No: 2305003007

```python

# === Simple LSTM Implementation using TensorFlow-Keras ===

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# --- Step 1: Prepare a simple sequence dataset ---
sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

X = []
y = []
for i in range(len(sequence) - 3):
    X.append(sequence[i:i+3])
    y.append(sequence[i+3])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Step 2: Build LSTM model ---
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

# --- Step 3: Compile model ---
model.compile(optimizer='adam', loss='mse')

# --- Step 4: Train the model ---
history = model.fit(X, y, epochs=200, verbose=0)

# --- Step 5: Test prediction ---
test_input = np.array([7, 8, 9]).reshape((1, 3, 1))
prediction = model.predict(test_input)

print("Input Sequence: [7, 8, 9]")
print("Predicted next number:", prediction[0][0])


# --- GRAPH 1: Training Loss Curve ---
plt.plot(history.history['loss'])
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --- GRAPH 2: Sequence and Prediction ---
plt.plot(sequence, marker='o', label="Original Sequence")
plt.scatter(len(sequence), prediction[0][0], color='red', label="Predicted Value", s=80)
plt.title("Sequence and Predicted Next Number")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

```

## OUTPUT:

### Predicted Value: 

<img width="691" height="82" alt="image" src="https://github.com/user-attachments/assets/29b1462f-daa8-4f97-9715-659008e000a7" />

### Training Loss Curve:

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/5103a510-b72a-4aa1-b0c8-e50d161e7108" />

### Sequence and Prediction:

<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/fda964f2-4e31-46a0-8084-d5157f6ba05b" />


## RESULT:
Thus, the program to implement a Simple LSTM neural network for sequence prediction has been successfully developed and executed.
