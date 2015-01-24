import pandas as pd
import numpy as np
import time
from datetime import datetime
from neural_network import NeuralNetwork

N_INPUTS = 25
N_OUTPUTS = 10
N_NEURONS = 100
N_LAYERS = 1
SAMPLE_SIZE = 10000
TEST_SIZE = 100
N_STEPS = 10
STEP_SIZE = 0.0001
STEP_DECAY = 0.5

def get_timestamp():
    raw = datetime.now()
    return str(raw.year) + str(raw.month) + str(raw.day) + "-" + str(raw.hour) + str(raw.minute)

print("COMP90051 - Statistical and Evolutionary Learning\n"
      "Project 2 - Neural Networks\n\n")

print("1. Loading data")
data = pd.read_csv("data/train.csv",header=None)
test = pd.read_csv("data/test-nolabel.csv",header=None,delimiter=" ")

print("2. Sampling data")
np.random.seed(1)
sample = np.random.choice(len(data.values),SAMPLE_SIZE + TEST_SIZE,replace=False)
i_train = sample[:SAMPLE_SIZE]
i_test = sample[SAMPLE_SIZE:]

sample_train = data.iloc[i_train]
sample_test = data.iloc[i_test]

print("3. Training model")

train_start = time.time()
nn = NeuralNetwork(N_INPUTS,N_OUTPUTS,N_NEURONS,N_LAYERS)
# nn.train(sample_train.values,N_STEPS,STEP_SIZE,STEP_DECAY,N_OUTPUTS)
nn.train(data.values,N_STEPS,STEP_SIZE,STEP_DECAY,N_OUTPUTS)
train_end = time.time()

print("4. Running predictions")

predict_start = time.time()

results = []
# for x in range(0,len(sample_test.values)):
#     inputs, outputs, label = nn.prepare_instance(sample_test.values[x],N_OUTPUTS)
#     prediction, max_score, activations = nn.predict(inputs)
#     error = float(prediction) - float(label)
#     results.append((prediction,label,max_score,activations,error**2))

for x in range(0,len(test.values)):
    inputs, outputs, label = nn.prepare_instance(test.values[x],N_OUTPUTS)
    prediction, max_score, activations = nn.predict(inputs)
    results.append((prediction,label))

predict_end = time.time()

print("5. Writing results")
results_out = pd.DataFrame(results,columns=['Prediction','Label'])
results_out.Prediction.to_csv("output-" + get_timestamp() + ".csv",index=None)

# # calculate rmse
# rmse = 0.0
# for result in results:
#     rmse += result[-1]
# rmse = rmse**0.5

# print("\n\nSTATISTICS\nTrain = %s seconds"
#       "\nPrediction %s seconds"
#       "\nTotal %s seconds"
#       "\nRMSE %s" % (train_end - train_start,predict_end - predict_start,predict_end-train_start,rmse))
