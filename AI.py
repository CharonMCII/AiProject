import os
import shutil
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Directory da cui controllare i file
directory = "percorso/alla/tua/directory"
output_directory = "percorso/alla/tua/directory/output"

# Creazione delle cartelle di output
output_good_dir = os.path.join(output_directory, "good")
output_bad_dir = os.path.join(output_directory, "bad")
output_queue_dir = os.path.join(output_directory, "queue")

os.makedirs(output_good_dir, exist_ok=True)
os.makedirs(output_bad_dir, exist_ok=True)
os.makedirs(output_queue_dir, exist_ok=True)

# Elenco dei file nella directory
files = os.listdir(directory)

# Generazione dei dati di esempio e spostamento dei file nelle cartelle appropriate
for file_name in files:
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            file_contents = file.read()
            if '0' in file_contents and '9' in file_contents:
                shutil.move(file_path, os.path.join(output_queue_dir, file_name))
                print(f"{file_name}: queue")
            elif '0' in file_contents:
                shutil.move(file_path, os.path.join(output_bad_dir, file_name))
                print(f"{file_name}: bad")
            elif '9' in file_contents:
                shutil.move(file_path, os.path.join(output_good_dir, file_name))
                print(f"{file_name}: good")

# Generazione dei dati di addestramento
num_good_files = len(os.listdir(output_good_dir))
num_bad_files = len(os.listdir(output_bad_dir))
num_queue_files = len(os.listdir(output_queue_dir))

num_samples = num_good_files + num_bad_files + num_queue_files
input_size = 10
x_train = np.random.random((num_samples, input_size))
y_train = np.concatenate([
    np.ones((num_good_files, 1)),
    np.zeros((num_bad_files, 1)),
    np.ones((num_queue_files, 1))
])

# Creazione del modello
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilazione del modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Esempio di predizione su nuovi dati
new_files = np.random.random((5, input_size))
predictions = model.predict(new_files)
print(predictions)
