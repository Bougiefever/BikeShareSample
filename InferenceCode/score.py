import json
import numpy as np
import os, glob
import argparse
import pickle
import math
from cntk.ops.functions import load_model

MODEL = ''
def unpickle(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    return data

def one_Hot_String(val_list, val, sep=' '):
    array = np.eye(len(val_list))[val_list.index(val)].astype(int)
    return sep.join(str(i) for i in array.tolist())

def init():
    '''Initialize the model'''
    global MODEL 
    MODEL = load_model('model.dnn')

def pre_process(input_data):
    '''Preprocess data to prep for ML'''
    #print(data)
    data = json.loads(input_data)
    time_bucket = data['time_bucket']
    day = data['day']
    dry_bulb_temp = data['dry_bulb_temp']
    relative_humidity = data['relative_humidity']
    hourly_wind_speed = data['hourly_wind_speed']
    station_id = data['station_id']

    days = ['Sun', 'Mon', 'Thu', 'Sat', 'Wed', 'Tue', 'Fri'] # unpickle('days.pkl')
    time_buckets = ['2AM-4AM', '4PM-6PM', '12PM-2PM', '10PM-12AM', '12AM-2AM', '6PM-8PM', '2PM-4PM', '4AM-6AM', '8PM-10PM', '6AM-8AM', '10AM-12PM', '8AM-10AM'] # unpickle('time_buckets.pkl')
    station_ids = ['67.0', '36.0', '46.0', '10.0', '47.0', '107.0', '58.0', '9.0', '88.0', '89.0', '133.0', '27.0', '80.0', '22.0', '39.0', '178.0', '42.0', '4.0', '115.0', '161.0', '102.0', '33.0', '84.0', '25.0', '131.0', '152.0', '51.0', '21.0', '23.0', '32.0', '94.0', '49.0', '54.0', '146.0', '179.0', '30.0', '96.0', '190.0', '8.0', '41.0', '110.0', '197.0', '93.0', '68.0', '145.0', '19.0', '105.0', '180.0', '87.0', '185.0', '119.0', '59.0', '73.0', '118.0', '77.0', '215.0', '16.0', '81.0', '11.0', '213.0', '98.0', '14.0', '195.0', '163.0', '104.0', '78.0', '176.0', '184.0', '76.0', '95.0', '43.0', '74.0', '6.0', '91.0', '169.0', '183.0', '100.0', '116.0', '124.0', '70.0', '7.0', '75.0', '173.0', '151.0', '15.0', '5.0', '139.0', '218.0', '150.0', '31.0', '24.0', '20.0', '3.0', '44.0', '130.0', '17.0', '56.0', '174.0', '142.0', '159.0', '71.0', '40.0', '177.0', '175.0', '160.0', '189.0', '63.0', '108.0', '141.0', '149.0', '137.0', '109.0', '37.0', '117.0', '90.0', '57.0', '121.0', '140.0', '208.0', '72.0', '143.0', '186.0', '85.0', '12.0', '217.0', '200.0', '126.0', '138.0', '170.0', '65.0', '97.0', '210.0', '135.0', '201.0', '171.0', '92.0', '219.0', '29.0', '136.0', '205.0', '196.0', '212.0', '207.0', '1.0', '70.0', 'ERROR.0', '55.0', '125.0', '60.0', '45.0', '53.0', '66.0', '39.0', '109.0', '38.0', '112.0', '42.0', '13.0', '99.0', '129.0', '114.0', '86.0', '48.0', '26.0', '122.0', '111.0', '123.0', '130.0', '64.0', '139.0', '103.0', '113.0', '131.0', '128.0', '120.0', '35.0', '69.0', '106.0', '61.0', '134.0', '50.0', '52.0', '82.0', '79.0', '132.0', '92.0', '162.0', '167.0', '192.0', '193.0', '194.0', '60.0', '52.0', '216.0', '209.0', '211.0', '214.0', '204.0', '202.0', '203.0', '199.0', '153.0'] #  unpickle('station_ids.pkl')

     # one hot encode day
    days_encoded = one_Hot_String(days, day)

    #one hot encode time bucket
    time_bucket_encoded = one_Hot_String(time_buckets, time_bucket)

    #one hot encode station id
    station_id_encoded = one_Hot_String(station_ids, station_id)

    weather = [dry_bulb_temp, relative_humidity, hourly_wind_speed]

    day_arr = np.asarray(days_encoded.strip().split(' '), dtype=np.float32)
    time_bucket_arr = np.asarray(time_bucket_encoded.strip().split(' '), dtype=np.float32)
    station_id_arr = np.asarray(station_id_encoded.strip().split(' '), dtype=np.float32)
    weather_arr = np.asarray(weather, dtype=np.float32)
    arr = np.concatenate((time_bucket_arr, day_arr, weather_arr, station_id_arr), axis=0)
    return arr

def predict(proc_data):
    '''returns prediction from processed data'''
    return MODEL.eval(proc_data)

def post_process(predictions):
    '''post processes predictions'''
    num = predictions[0][0]
    val = math.ceil(num)
    result = json.dumps({'raw_result': str(num), 'prediction': val })
    return result

def run(input_data):
    '''Entry point for ML operationalization'''
    try:
        proc_data = pre_process(input_data)
        predictions = predict(proc_data)
        post_proc = post_process(predictions)
    except Exception as exc:
        return (str(exc))
    return post_proc

if __name__ == "__main__":
    init()
    arr = "|label 2 |features 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0.46919431279620855 0.3103448275862069 0.3380281690140845 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
    arr = arr.strip().split(' ')
    arr = arr[3:]
    arr = np.asarray(arr, dtype=np.float32)
    
    pred = predict(arr)
    print("Prediction: ", pred, " Actual was 2")

    dict = {
        'time_bucket': '12AM-2AM', 
        'day': 'Sun', 
        'dry_bulb_temp': 0.450236967,
        'relative_humidity': 0.873563218,
        'hourly_wind_speed': 0.436619718,
        'station_id': '67.0'
    }
    json_data = json.dumps(dict, )
    result = run(json_data)
    print(result)
    #data = { 'time_bucket': '12AM-2AM', 'day': 'Sun', 'dry_bulb_temp': 0.450236967, 'relative_humidity': 0.873563218, 'hourly_wind_speed': 0.436619718, 'station_id': '67.0'}


    #arrayresult = pre_process(data)
    # raw_data = ['12AM-2AM','Sun',0.450236967,0.873563218,0.436619718,'67.0']
    # target = 2
    # data = pre_process(raw_data)
    # prediction = predict(data)
    # result = post_process(prediction)
    # print(result, target)

    # test_data = ['2PM-4PM','Mon',0.45971564,0.626436782,0.112676056,'6.0']
    # test = run(test_data)
    # print(test)