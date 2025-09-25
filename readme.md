This is the code repository for the paper "Using reinforcement learning to solve vehicle routing problems with dynamic customers"

This code inherits from the code accompanying the paper "Learning TSP Requires Rethinking Generalization" so thanks to the authors for making their code accessible. The main changes come in the form of 4 new simulators for simulating the following 4 problems.

- Partially Dynamic Travelling Repairman Problem (PDTRP)
- Partially Dynamic Capacitated Travelling Repairman Problem (PDCTRP)
- Partially Dynamic Travelling Repairman Problem with Time Windows (PDTRPTW)
- Partially Dynamic Capacitated Travelling Repairman Problem with Time Windows (PDCTRPTW)

[Confusingly, in this code base, the acronym PDCVRP(T) is used instead of PDCTRP(TW). This is because I changed my mind about the naming scheme, it was originally Partially Dynamic Capacitated Vehicle Rouitng Problem (with Time Windows), and I haven't got around to changing all the references in the code to match yet.]

There is also the dynamic attention model which is a version of the original attention model from "Attention, Learn to Solve Routing Problems!" that has been adapted to work with each of the 4 dynamic problems and contains other features like the arrival LSTM. 

The main training script has weights and biases built in and makes nice animations of validation runs for each of the problems. 

The repository is a bit rudamentory at the moment but I hope to do a big code refactoring soon that will make it more user friendly.

I have tried to do my best to make it so that the experimental results and ablation results from the paper are easy to recreate. The main places to look are the example_scripts folder and the examples.ipynb notebook.

The datasets and models used to generate the results are in named folders. To generate the results using these is a case of specifying the right arguments in the testing.py script. I've included an example testing and training script that will hopefully provide enough guidance. 

I've used conda to generate environment.yml file that should create a conda environment to run the code in this repo in. It may be necessary to install the correct version of pytorch to match the hardware on your system. 

