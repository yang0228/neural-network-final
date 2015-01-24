import random
from neural_network import NeuralNetwork

class GeneticAlgorithmUtility:

    def __init__(self,
                 n_inputs,n_outputs,
                 max_neurons,min_neurons,
                 max_layers,min_layers,
                 max_steps,min_steps,
                 max_step_size,min_step_size,
                 max_step_decay,min_step_decay,
                 max_weight_bound,min_weight_bound,
                 max_bias_bound,min_bias_bound
    ):
        """
        Constructor
        :param max_neurons:
        :param min_neurons:
        :param max_layers:
        :param min_layers:
        :param max_steps:
        :param min_steps:
        :param max_step_size:
        :param min_step_size:
        :param max_step_decay:
        :param min_step_decay:
        :return:
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        self.max_step_decay = max_step_decay
        self.min_step_decay = min_step_decay
        self.max_weight_bound = max_weight_bound
        self.min_weight_bound = min_weight_bound
        self.max_bias_bound = max_bias_bound
        self.min_bias_bound = min_bias_bound

    def random_network(self):
        """
        Produces a random NN
        :return:
        """
        return NeuralNetwork(
            self.n_inputs,
            self.n_outputs,
            random.randint(self.min_neurons,self.max_neurons),
            random.randint(self.min_layers,self.max_layers),
            random.randint(self.min_steps,self.max_steps),
            random.uniform(self.min_step_size,self.max_step_size),
            random.uniform(self.min_step_decay,self.max_step_decay),
            random.uniform(self.min_weight_bound,self.max_weight_bound),
            random.uniform(self.min_bias_bound,self.max_bias_bound)
        )

    def encode_network(self,nn):
        """
        Produces a genotype for the network
        """
        return [nn.n_inputs,nn.n_outputs,nn.n_neurons,nn.n_hidden_layers,nn.n_steps,nn.step_size,nn.step_decay,nn.weight_bound,nn.bias_bound]

    def decode_network(self,g):
        """
        Produces a network (phenotype) from a genotype
        """
        return NeuralNetwork(g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8])

    def combine(self,ga,gb):
        """
        Combines the genotypes of two networks at a random point
        Note: the crossover must happen past the fixed entries (inputs/outputs)
        """
        crossover = random.randint(2,len(ga)-1)
        gc = ga[0:crossover]
        gc += gb[crossover:]
        return gc

    def mutate(self,gc):
        """
        Mutates the offspring genotype
        Note: we let the mutation go past the array to allow for no mutation
        :param gc:
        :return:
        """
        mutation = random.randint(2,len(gc))

        # n_neurons
        if mutation == 2:
            gc[2] = random.randint(self.min_neurons,self.max_neurons)

        # n_hidden_layers
        elif mutation == 3:
            gc[3] = random.randint(self.min_layers,self.max_layers)

        # n_steps
        elif mutation == 4:
            gc[4] = random.randint(self.min_steps,self.max_steps)

        # step_size
        elif mutation == 5:
            gc[5] = random.uniform(self.min_step_size,self.max_step_size)

        # step_decay
        elif mutation == 6:
            gc[6] = random.uniform(self.min_step_decay,self.max_step_decay)

        # weight bound
        elif mutation == 7:
            gc[7] = random.uniform(self.min_weight_bound,self.max_weight_bound)

        # bias bound
        elif mutation == 8:
            gc[8] = random.uniform(self.min_bias_bound,self.max_bias_bound)

        return gc