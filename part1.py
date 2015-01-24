"""
Project 2 - Artificial Neural Networks & Genetic Algorithms

Part 1 - Artificial Neural Networks
"""
import pandas as pd
import time
import numpy as np
from utility import Utility
from neural_network import NeuralNetwork

# model params
N_INPUTS = 25
N_OUTPUTS = 10
N_NEURONS = 100
N_LAYERS = 1
SAMPLE_SIZE = 20000
TEST_SIZE = 20000
N_STEPS = 3
STEP_SIZE = 0.001
STEP_DECAY = 0.5
WEIGHT_BOUND = 0.02
BIAS_BOUND = 0.05
MOMENTUM = 0.1

u = Utility()
u.print_header()
print("Part 1 - Artificial Neural Networks\n")

u.print_step("Loading data")
data = pd.read_csv("data/train.csv",header=None)
test = pd.read_csv("data/test-nolabel.csv",header=None,delimiter=" ")

# LEFT IN FOR IF YOU WANT TO TEST ON THE TRAINING SET
# u.print_step("Sampling data")
# samples = np.random.choice(len(data.values),SAMPLE_SIZE+TEST_SIZE,replace=False)
# data = data.iloc[samples[:SAMPLE_SIZE]]
# data_test = data.iloc[samples[SAMPLE_SIZE:]]

u.print_step("Training model, please wait...")
train_start = time.time()
nn = NeuralNetwork(N_INPUTS,N_OUTPUTS,N_NEURONS,N_LAYERS,N_STEPS,STEP_SIZE,STEP_DECAY,WEIGHT_BOUND,BIAS_BOUND,MOMENTUM)
nn.train(data.values)
train_end = time.time()

u.print_step("Running predictions, please wait...")
predict_start = time.time()
predictions = nn.predict_all(test.values)
predict_end = time.time()

time_train = train_end - train_start
time_predict = predict_end - predict_start
time_total = time_train+time_predict

u.print_step("Writing submission file...")
output = pd.DataFrame(predictions,columns=["Predictions","Labels"])
output["Predictions"].to_csv("studentno_name_stage1.txt",header=None,index=None)

print("DONE")

# print stats
print("Train = %s s" % time_train)
print("Predict = %s s" % time_predict)
print("Total = %s s" % time_total)
