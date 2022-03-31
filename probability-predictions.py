"""
    # PROBABILITY PREDICTIONS
    # 1)
    # Cold days encoded as 1
    # Hot  days encoded as 0
    # 
    # 2) First day 80% chance being cold
    # 3) cold day has 30% chance of being followed by hot day
    # 4) hot day  has 20% chance of being followed by cold day
    # 5) on each day the temperature is normally distributed with mean and standard deviation:
    #       –  0 and  5 on cold days
    #       – 15 and 10 on  hot days
"""


import tensorflow_probability as tfp
import tensorflow as tf
import jstring as j


j.clear()
print(tf.__version__)
print(tfp.__version__)

tfd = tfp.distributions #shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # Refer to #2) in intro
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.1, 0.9]]) #[cold:[cold, hot], hot[cold, hot]]
# two states (0:cold,1:hot) and both states have chances for being followed by one of the two stats
observation_distribution = tfd.Normal(loc=[0., 25.], scale=[15., 25.]) #refer to #5)
# loc=[cold, hot], scale[cold, hot]
# loc is average temperature, scale is standard deviation 
# 0. the dot makes an float out of the 0 —— [0.] type float while [0] type int

### Creating the model
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

### Get results
mean = model.mean()

### print results
with tf.compat.v1.Session() as sess:
    j.clear()
    days = ["Mo","Tu","We","Th","Fr","Sa","Su"]
    for name in days:
        print(name, end="\t")
    print()
    for day in mean.numpy():
        print(round(day,1),end="°\t")
    print("\n")
  



