import pandas as pd
import numpy as np
import time
from datetime import datetime
from neural_network import NeuralNetwork

N_INPUTS = 25
N_OUTPUTS = 10
N_NEURONS = 100
N_LAYERS = 1
SAMPLE_SIZE = 20000
TEST_SIZE = 1000
N_STEPS = 3
STEP_SIZE = 0.0001
STEP_DECAY = 0.5

WEIGHT_BOUND = 0.02
BIAS_BOUND = 0.05

def get_timestamp():
    raw = datetime.now()
    return str(raw.year) + str(raw.month) + str(raw.day) + "-" + str(raw.hour) + str(raw.minute)

print("Loading data")
data = pd.read_csv("data/train.csv",header=None)

print("Sampling data")
np.random.seed(1)
samples = np.random.choice(len(data.values),SAMPLE_SIZE+TEST_SIZE,replace=False)
data_train = data.iloc[samples[:SAMPLE_SIZE]]
data_test = data.iloc[samples[SAMPLE_SIZE:]]

print("Training...")
train_start = time.time()
nn = NeuralNetwork(N_INPUTS,N_OUTPUTS,N_NEURONS,N_LAYERS,N_STEPS,STEP_SIZE,STEP_DECAY,WEIGHT_BOUND,BIAS_BOUND)
nn.train(data_train.values)
train_end = time.time()

print("Predicting...")
predict_start = time.time()
results = []
count = 0.0
correct = 0.0
rmse = 0.0
for x in range(0,len(data_test.values)):
    inputs, outputs, label = nn.prepare_instance(data_test.values[x],N_OUTPUTS)
    prediction, max_score, activations = nn.predict(inputs)
    error = float(prediction) - float(label)
    results.append((prediction,label,max_score,activations,error**2))
    count += 1.0
    rmse += error**2
    if error == 0.0:
        correct += 1.0
predict_end = time.time()

accuracy = (correct / count)*100
rmse = rmse**0.5
time_train = train_end - train_start
time_predict = predict_end - predict_start
time_total = time_train+time_predict

print("Accuracy = %s" % accuracy)
print("RMSE = %s" % rmse)
print("Train = %s" % time_train)
print("Predict = %s" % time_predict)
print("Total = %s" % time_total)
#print(results)

print(nn.step_size)


# print("Training model")
# nn = NeuralNetwork(N_INPUTS,N_OUTPUTS,N_NEURONS,N_LAYERS)

# print("COMP90051 - Statistical and Evolutionary Learning\n"
#       "Project 2 - Neural Networks\n\n"
#
# print("1. Loading data")
# data = pd.read_csv("data/train.csv",header=None)
# test = pd.read_csv("data/test-nolabel.csv",header=None,delimiter=" ")
#
# print("2. Sampling data")
# np.random.seed(1)
# sample = np.random.choice(len(data.values),SAMPLE_SIZE + TEST_SIZE,replace=False)
# i_train = sample[:SAMPLE_SIZE]
# i_test = sample[SAMPLE_SIZE:]
#
# sample_train = data.iloc[i_train]
# sample_test = data.iloc[i_test]
#
# print("3. Training model")
#
# train_start = time.time()
# nn = NeuralNetwork(N_INPUTS,N_OUTPUTS,N_NEURONS,N_LAYERS)
# # nn.train(sample_train.values,N_STEPS,STEP_SIZE,STEP_DECAY,N_OUTPUTS)
# nn.train(data.values,N_STEPS,STEP_SIZE,STEP_DECAY,N_OUTPUTS)
# train_end = time.time()
#
# print("4. Running predictions")
#
# predict_start = time.time()
#
# results = []
# # for x in range(0,len(sample_test.values)):
# #     inputs, outputs, label = nn.prepare_instance(sample_test.values[x],N_OUTPUTS)
# #     prediction, max_score, activations = nn.predict(inputs)
# #     error = float(prediction) - float(label)
# #     results.append((prediction,label,max_score,activations,error**2))
#
# for x in range(0,len(test.values)):
#     inputs, outputs, label = nn.prepare_instance(test.values[x],N_OUTPUTS)
#     prediction, max_score, activations = nn.predict(inputs)
#     results.append((prediction,label))
#
# predict_end = time.time()
#
# print("5. Writing results")
# results_out = pd.DataFrame(results,columns=['Prediction','Label'])
# results_out.Prediction.to_csv("output-" + get_timestamp() + ".csv",index=None)
#
# # # calculate rmse
# # rmse = 0.0
# # for result in results:
# #     rmse += result[-1]
# # rmse = rmse**0.5
#
# # print("\n\nSTATISTICS\nTrain = %s seconds"
# #       "\nPrediction %s seconds"
# #       "\nTotal %s seconds"
# #       "\nRMSE %s" % (train_end - train_start,predict_end - predict_start,predict_end-train_start,rmse))
