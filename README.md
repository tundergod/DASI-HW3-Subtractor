# DASI-HW3-Subtractor

## Test Environment
- Programming Language: Python 3.5.2 (pip3 distribution)
- Operating System: Ubuntu 16.04.3 LTS

## Platform
- using Keras platform, with TensorFlow backend : Keras 2.1.4, TensorFlow 1.5.0 (cpu only)

## Usage
- python3 main.py

## Explanation
- Model : a simple seq2seq model
- Data input : ['123+123','8+9    ','23+123 ','99+666 ']
- Model saving and loading: 
  - h5py module to save, keras module to load, model format : "model.h5"
  - check before training, if "model.h5" exists, load and train, build new model if not
  - save after training



