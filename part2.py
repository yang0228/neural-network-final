"""
Project 2 - Artificial Neural Networks & Genetic Algorithms

Part 2 - Genetic Algorithm
"""
import time
import heapq
import numpy as np
import pandas as pd
from utility import Utility
from ga import GeneticAlgorithmUtility

# model parameters
TRAIN_SIZE = 20000
TEST_SIZE = 20000
N_GENERATIONS = range(0,10)
MAX_POPULATION = 4
MAX_OFFSPRING = 4
N_INPUTS = 25
N_OUTPUTS = 10
MAX_NEURONS = 100
MIN_NEURONS = 20
MAX_LAYERS = 3
MIN_LAYERS = 1
MAX_STEPS = 3
MIN_STEPS = 1
MAX_STEP_SIZE = 0.5
MIN_STEP_SIZE = 0.1
MAX_STEP_DECAY = 0.9
MIN_STEP_DECAY = 0.1
MIN_WEIGHT_BOUND = 0.01
MAX_WEIGHT_BOUND = 0.02
MIN_BIAS_BOUND = 0.01
MAX_BIAS_BOUND = 0.02
MAX_MOMENTUM = 0.9
MIN_MOMENTUM = 0.1

# set up the utility class
u = Utility()

# set up the GA utility class
gau = GeneticAlgorithmUtility(
    N_INPUTS,
    N_OUTPUTS,
    MAX_NEURONS,
    MIN_NEURONS,
    MAX_LAYERS,
    MIN_LAYERS,
    MAX_STEPS,
    MIN_STEPS,
    MAX_STEP_SIZE,
    MIN_STEP_SIZE,
    MAX_STEP_DECAY,
    MIN_STEP_DECAY,
    MAX_WEIGHT_BOUND,
    MIN_WEIGHT_BOUND,
    MAX_BIAS_BOUND,
    MIN_BIAS_BOUND,
    MAX_MOMENTUM,
    MIN_MOMENTUM
)

# print header
u.print_header()
print("Part 2 - Genetic Algorithm\n")

# load data
u.print_step("Loading data")
data = pd.read_csv("data/train.csv",header=None)
test = pd.read_csv("data/test-nolabel.csv",header=None,delimiter=" ")

# sample data
u.print_step("Sampling data")
data_train = data.iloc[np.random.choice(len(data.values),TRAIN_SIZE,replace=False)]
data_test = data.iloc[np.random.choice(len(data.values),TEST_SIZE,replace=False)]

# setup population
u.print_step("Setting up initial population")
population = [gau.random_network() for x in range(0,MAX_POPULATION)]
leaderboard = []
max_fitness = 0.0
min_fitness = np.infty
fitness_scores_at_gen = []

total_start = time.time()
u.print_step("Running simulation\n")

# start loop
for g in N_GENERATIONS:

    print("GENERATION %d\n" % g)

    total_fitness = 0.0
    fitnesses = []
    genotypes = []

    for p in range(0,MAX_POPULATION):

        nn = population[p]
        genotype = gau.encode_network(nn)

        print(genotype)
        train_start = time.time()
        nn.train(data_train.values)
        train_end = time.time()
        score, rmse = nn.predict_fitness(data_test.values)
        predict_end = time.time()
        print("Fitness = %s, RMSE = %s Train = %s, Predict = %s\n" % (score,rmse,train_end-train_start,predict_end-train_end))

        # update the max records
        if score > max_fitness:
            max_fitness = score
        if rmse < min_fitness:
            min_fitness = rmse

        total_fitness += score

        # add to priority queue (the inverse, so it is properly stacked)
        # heapq.heappush(leaderboard,(-fitness,genotype))
        heapq.heappush(leaderboard,(rmse,genotype))
        fitnesses.append(score)
        genotypes.append(genotype)

    # update the roulette with the probabilities
    probabilities = []
    for i in range(0,len(fitnesses)):
        probabilities.append(fitnesses[i]/total_fitness)

    # fill the offspring population
    print("COMBINATION/MUTATION\n")
    offspring = 0
    while offspring < MAX_OFFSPRING:

        # randomly select 2 parents
        parents = np.random.choice(len(genotypes),size=2,replace=False,p=probabilities)

        ga = genotypes[parents[0]]
        gb = genotypes[parents[1]]

        # combine their genotypes, mutate the offspring
        combination = gau.combine(ga,gb)
        mutation = gau.mutate(combination)
        print(mutation)

        offspring += 1

        # train the child
        nn = gau.decode_network(mutation)
        train_start = time.time()
        nn.train(data_train.values)
        train_end = time.time()

        # predictions...
        score, rmse = nn.predict_fitness(data_test.values)
        predict_end = time.time()
        print("Fitness = %s, RMSE = %s Train = %s, Predict = %s\n" % (score,rmse,train_end-train_start,predict_end-train_end))

        # update stats
        if score > max_fitness:
            max_fitness = score
        if rmse < min_fitness:
            min_fitness = rmse

        # add to priority queue
        # heapq.heappush(leaderboard,(-fitness,mutation))
        heapq.heappush(leaderboard,(rmse,mutation))

    # take everyone out of the arrays
    population = []

    for i in range(0,MAX_POPULATION):
        fitness, genotype = heapq.heappop(leaderboard)
        population.append(gau.decode_network(genotype))

    # empty the priority heap
    leaderboard = []

    # update stats, for charts
    fitness_scores_at_gen.append((score,rmse))

    print("GENERATION COMPLETE\n")

# at this point, the strongest genotype is the first one in the population array
fittest = population[0]
genotype = gau.encode_network(fittest)
print("GA COMPLETE\nStrongest candidate is")
print(genotype)
print("Fitness = %s RMSE = %s" % (max_fitness , min_fitness))

# retrain
u.print_step("Retraining candidate with full 60K...")
train_start = time.time()
fittest.train(data.values)
train_end = time.time()

# run predictions
u.print_step("Predicting unlabeled data...")
predict_start = time.time()
predictions = fittest.predict_all(test.values)
predict_end = time.time()

print("Train = %s, Predict = %s\n" % (train_end-train_start,predict_end-train_end))

# write submission
u.print_step("Writing submission file...")
results = pd.DataFrame(predictions,columns=['Prediction','Label'])
results['Prediction'].to_csv("studentno_name_stage2.txt",header=False,index=False)

total_end = time.time()

print("DONE")

print("Total runtime = %s" % (total_end-total_start))

# Left in to look at convergence
# print(fitness_scores_at_gen)
# convergence = pd.DataFrame(fitness_scores_at_gen,columns=['acc','rmse'])
# convergence.to_csv("convergence-" + u.get_timestamp() + ".csv")
