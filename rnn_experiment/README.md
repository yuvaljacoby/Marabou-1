Self compare experiments compare between gurobi (with specific configuration) and random.
To run we use rnn_experiment/self_compare/experiment.py and the controlled
experiment there.
the command is:
PYTHONPATH=. python3 rnn_experiment/self_compare/experiment.py controlled model_20classes_rnn4_fc32_epoc
40.h5 points_0_2_200.pkl 0 2>&1 | tee logs/rnn4_controlled_0.log


i.e.
file - rnn_experiment/self_compare/experiment.py 
args:
"controlled" - to indicated which type of experiment
path_to_h5 - which rnn to use
path_to_points - list of points to use (array where each cell is array of shape
40)
0 - in which index to start running in the list points (in order to enable
parallel execution) 

the "tee" trick is to display the outputs and save them to the file.
