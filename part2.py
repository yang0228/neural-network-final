import time
import heapq
import numpy as np
import pandas as pd
import utilities as u
from ga import GeneticAlgorithmUtility

TRAIN_SIZE = 5000
TEST_SIZE = 5000

N_GENERATIONS = range(0,10)
MAX_POPULATION = 4
MAX_OFFSPRING = 4

N_INPUTS = 25
N_OUTPUTS = 10
MAX_NEURONS = 100
MIN_NEURONS = 1
MAX_LAYERS = 5
MIN_LAYERS = 1
MAX_STEPS = 5
MIN_STEPS = 1
MAX_STEP_SIZE = 0.1
MIN_STEP_SIZE = 0.0001
MAX_STEP_DECAY = 0.9
MIN_STEP_DECAY = 0.1

MIN_WEIGHT_BOUND = 0.01
MAX_WEIGHT_BOUND = 0.02
MIN_BIAS_BOUND = 0.01
MAX_BIAS_BOUND = 0.02

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
    MIN_BIAS_BOUND
)

print("COMP90051 - Statistical and Evolutionary Learning\n"
      "Project 2 - Neural Networks\n"
      "Part 2 - Genetic Algorithm\n\n"
      )

print("Loading data...")
data = pd.read_csv("data/train.csv",header=None)
test = pd.read_csv("data/test-nolabel.csv",header=None,delimiter=" ")
# np.random.seed(1)
data_train = data.iloc[np.random.choice(len(data.values),TRAIN_SIZE,replace=False)]
data_test = data.iloc[np.random.choice(len(data.values),TEST_SIZE,replace=False)]

# setup the population
population = [gau.random_network() for x in range(0,MAX_POPULATION)]
offspring_population = []
leaderboard = []

print("STARTING SIMULATION\n")

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
        fitness = nn.predict_fitness(data_test.values)
        predict_end = time.time()
        print("Fitness = %s, Train = %s, Predict = %s\n" % (fitness,train_end-train_start,predict_end-train_end))

        total_fitness += fitness

        # add to priority queue (the inverse, so it is properly stacked)
        heapq.heappush(leaderboard,(-fitness,genotype))
        fitnesses.append(fitness)
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
        fitness = nn.predict_fitness(data_test.values)
        predict_end = time.time()
        print("Fitness = %s, Train = %s, Predict = %s\n" % (fitness,train_end-train_start,predict_end-train_end))

        # add to priority queue
        heapq.heappush(leaderboard,(-fitness,mutation))

    # take everyone out of the arrays
    population = []

    for i in range(0,MAX_POPULATION):
        fitness, genotype = heapq.heappop(leaderboard)
        population.append(gau.decode_network(genotype))

    leaderboard = []

    print("GENERATION COMPLETE\n")

# at this point, the strongest genotype is the first one in the population array
fittest = population[0]
genotype = gau.encode_network(fittest)
print("GA COMPLETE\nStrongest candidate is")
print(genotype)

print("\nPredicting unlabeled data")
predict_start = time.time()
predictions = fittest.predict_all(test.values)
predict_end = time.time()

print("Predict = %s" % (predict_end - predict_start))

print("Writing output")
results = pd.DataFrame(predictions,columns=['Prediction','Label'])
results['Prediction'].to_csv("output" + u.get_timestamp() + ".txt",header=False,index=False)

print("DONE")
