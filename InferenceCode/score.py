import json
import numpy as np
import os, glob
import argparse
import cv2
from cntk.ops.functions import load_model
import matplotlib.pyplot as plt

MODEL = ''

def init():
    '''Initialize the model'''
    global MODEL 
    MODEL = load_model('./model_files/model.dnn')

def pre_process(data):    
    '''Normalize pixel information and split into 256x256'''
    return img

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