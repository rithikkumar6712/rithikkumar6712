# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Function to display an image and its predicted label
def display_prediction(index):
    image = x_test[index].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    prediction = model.predict(x_test[index].reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)
    print(f"Predicted Digit: {predicted_label}")

# Display the prediction for a random test image
import random
random_index = random.randint(0, len(x_test) - 1)
display_prediction(random_index)



output


...
Epoch 10/10
48000/48000 [==============================] - 16s 342us/sample - loss: 0.0380 - accuracy: 0.9886 - val_loss: 0.0694 - val_accuracy: 0.9792

10000/10000 [==============================] - 1s 81us/sample - loss: 0.0629 - accuracy: 0.9815
Test Accuracy: 98.15%

<Displaying a random test image>
Predicted Digit: 3
