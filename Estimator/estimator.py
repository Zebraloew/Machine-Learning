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

### the core of the machine learning code
def input_fn(features, labels, training=True, batch_size=256):
    # Convert inputs to dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle if training
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

### feature_columns setup
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

### classifier
classifier = tf.estimator.DNNClassifier(    # DNN Deep Neural Network
    feature_columns = my_feature_columns,
    hidden_units=[30,10],                   # two hidden layers of 30 and 10 nodes respectively
    n_classes=3)                            # the model must choose between 3 classes

### training
classifier.train(  
    input_fn=lambda: input_fn(train, train_y, training=True), # lambda is a one line function. to oversimplify.
    steps=5000
    )

### evalute
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print("Evaluation".upper)
print("----------")
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


### predict
## test case
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
## prediction function
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))

## print prediction
for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))


### end
print("\n——END——\n")