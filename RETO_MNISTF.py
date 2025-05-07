import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import AdamW

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(255, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(255, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(255, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')
])

#model.compile(
    #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #loss='sparse_categorical_crossentropy',
    #metrics=['accuracy']
#)

model.compile(
    optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Fashion MNIST: Training vs Validation Accuracy')
plt.legend()
plt.show()

predictions = model.predict(x_test)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array[i])
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(
        f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}",
        color=color
    )

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plot_image(i, predictions, y_test, x_test)
plt.tight_layout()
plt.show()