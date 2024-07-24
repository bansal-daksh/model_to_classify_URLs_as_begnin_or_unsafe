import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import SGD # type: ignore
from sklearn.model_selection import train_test_split

// test


dataset = pd.read_csv('/Users/devmittal/Downloads/data/new_features/urls_final_complete.csv',low_memory=False, na_values='')

X = dataset.drop('URL_Type_obf_Type', axis=1)
y = dataset['URL_Type_obf_Type']

label_map = {'benign': 0, 'phishing': 1, 'malware': 1, 'defacement': 1, 'spam': 1}
y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = tf.convert_to_tensor(X_train, dtype = tf.float32)



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
