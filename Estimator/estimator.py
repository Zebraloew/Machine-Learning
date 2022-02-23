### this is a whole new project but will use the same environment as s.py
### based on this tutorial:
### https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/core-learning-algorithms-classification
### which refers to this site:
### https://www.tensorflow.org/tutorials/estimator/premade
### Jay 23.2.2022 – HH

import tensorflow as tf
import pandas as pd

### beginning
print(10*"•\n" + "FLOWERS" + "\n" + 7*"–" + "\n")

### define names for data
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

### load data
train_path = tf.keras.utils.get_file( "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path  = tf.keras.utils.get_file( "iris_test.csv",     "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

### put data into pandas structure
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test  = pd.read_csv(test_path,  names=CSV_COLUMN_NAMES, header=0)

### pop the layer column

train_y = train.pop('Species')
test_y  =  test.pop('Species')

### show first five entries in console
print(train.shape)

### the core of the mchine learning code
def input_fn(features, labels, training=True, batch_size=256):
    # Convert inputs to dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle if training
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


### end
print("\n END \n")