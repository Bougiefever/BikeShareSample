import json
import numpy as np
import os, glob
import argparse
from cntk.ops.functions import load_model

MODEL = ''

def init():
    '''Initialize the model'''
    global MODEL 
    MODEL = load_model('model.dnn')

def pre_process(data):    
    '''Preprocess data to prep for ML'''
    return data

def predict(proc_data):
    '''returns prediction from processed data'''
    return MODEL.eval(proc_data)

def post_process(predictions):
    '''post processes predictions'''
    return predictions

def run(input_data):
    '''Entry point for ML operationalization'''
    try:
        proc_data = pre_process(input_data)
        predictions = predict(proc_data)
        post_proc = post_process(predictions)
    except Exception as exc:
        return (str(exc))
    return json.dumps(post_proc)

if __name__ == "__main__":
    init()
    arr = "0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0.3696682464454976 0.6436781609195402 0.29577464788732394 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    arr = arr.split(' ')
    arr = np.asarray(arr, dtype=np.float32)
    
    pred = MODEL.eval( arr )
    print(pred)
