#### Tweets Classification and Machine Learning Experiments Tracking

This is and example of experiments registry using MLFlow. It consists on a Bi-LSTM for Tweets Classification implemented on Tf-Keras. The main intention of this repo is to illustrate the registry of experiments where parameters are modified as well as metrics logging. For cloning this repo and install mlflow run:

`git clone https://github.com/saulhazelius/keras-mlflow-sentences-classification.git`

`pip install mlflow`

Then, for running an example execute the next command in the repository directory:

`mlflow run -P hidden_units=32 -P dropout=0.1`

This is an example for an experiment that specifies 32 hidden units in the architecture and a dropout of 0.1.
In addition, the execution of the experiments are performed in a conda virtual environment where the dependencies are specified in the `train/conda.yaml` file. Finally, you can retrieve the metrics and experiment artifacts launching the dashboard with,

`mlflow ui`

and then opening it in the host http://127.0.0.1:5000/ .
