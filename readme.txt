##############################################################
Project 2 - Artificial Neural Networks & Genetic Algorithms

Please ensure that the data is in the data/ directory for the python scripts to run.

The scripts can be run directly with:

    python part1.py
    python part2.py

Both scripts will output their respective predictions in the working directory.

Caution: part2(GA) takes quite a while to run, 30-60min

Please note that the Pandas library is required to run the project, as it handles the file input/output.
#############################################################

1. Overview
ANN using the standard feed-forward and back-propagation methodologies with various optimisations
to produce predictions on unlabelled instances for Part 1. In Part 2, we extended our model to a Genetic Algorithm (GA) in
order to learn the optimal parameters for use in predictions. Our results illustrate that the ANN and GA approaches do indeed
mimic natural and biological processes through the reduction of classification error, hence demonstrating that the network has
the capacity to learn.

2. During the development of the model, we experimented with a number of optimisation techniques to improve performance,
namely:
	1)learning rate decay
	2)momentum
	3)dropout regularisation
	4)Genetic Algorithm

3. Evaluation
We use a blended approach to evaluate model performance, in the form of a fitness accuracy (positively
predicted labels, divided by total predictions) and Root Mean Squared Error. In doing this, we capture the fitness for use in
calculating selection probabilities(GA), as well as heavily penalising models with high RMSE.
