# Reinforcement Learning Examples
This directory contains Reinforcement Learning (RL) examples from 
[mattmcd](https://github.com/mattmcd).

# Installation
[pipenv](https://github.com/pypa/pipenv) is used for Python package management using
a base install of Python 3.7.  [TensorFlow](https://www.tensorflow.org) is currently 
installed via `pipenv run pip install tensorflow` rather than the Pipfile due to 
a dependency error whereby pipenv tries to install the Python 2 functools32 library.

Also installed edgetpu to try out environment image pre-processing into embedding 
vector using the [Coral Edge TPU](https://coral.ai/).  This required checking out the
[edgetpu](https://github.com/google-coral/edgetpu) github repo to build the 
python library.