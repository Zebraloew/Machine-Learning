### In VS Code Studio — you can set up the right environment as follows:
### press: option + shift + P
### write: "Python: Select Interpreter"
### py36 
### (This is my Python 3.6 legacy Conda environment. 
###  with a newer version, tensor flow does not run properly on El Capitan.)
### Then in bash: go to the conda environment
### deactivate
### conda deactivate
### conda activate py36

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Compatibility modules — not sure if needed
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

### gives error if on latest version – if running on el capitan
### so use legacy environment: py36 — python 3.6
import tensorflow as tf

### print intro
intro = " tensor flow tutorial program ".upper()
intro_len = len(intro)
print(5*"\n" + intro_len*"=")
print(intro)
print(intro_len*"=")


### import dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
print(5*"\n"+"Head of training data – First five passengers of the Titanic\n".upper() + 96*"–")
print(dftrain.head())
### extract "label" – the information the Neural network should find
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
#print("\n" , y_eval.head())
#print("\nData Summary: \n",dftrain.describe())
#print("\nShape: ",dftrain.shape)
#plt.hist(dftrain.age,bins=50)
#plt.show()

### testing numpy module
a = np.array([1,2,3])
#print(a)

### Setting up columns for tf

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #gets list of unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

### Testing columns
#print(3*"\n" + "Column Vocabulary\n" + 20*"—")
#for i in range(len(feature_columns)):
#    print(feature_columns[i],"\n")
print("\n"+"Unique entries in Column 'Embark Town': ",dftrain['embark_town'].unique())
### Actual Tensor Flow Code

def make_input_fn(data_df, label_df, num_epochs=20, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

### Tensor Flow in use
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) #create the estimator
linear_est.train(train_input_fn)                                            #train it
result = linear_est.evaluate(eval_input_fn)                                 #get metrics of model
prediction = list(linear_est.predict(eval_input_fn))

### Print some results finally
#clear_output() #where does this come from?
print(5*"\n")
print("Accuracy:\t",result['accuracy'])
print("Predictions:\n")
for i in range(5):
    print(33*". ")
    print(dfeval.loc[i])
    print("")
    print(f"Prediction for passenger {i} survival:\t{prediction[i]['probabilities'][1]}")
    status = "…"
    if y_eval.loc[i]:
        status = "survived"
    else:
        status = "died"
    print(f"And they actually {status}.\n\n")


### Casual print before exit
print("\n — THE END — \n")
