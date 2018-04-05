from __future__ import print_function
from azure.storage.blob import BlockBlobService
import os
import cntk as C
import numpy as np
import scipy.sparse
import sys
import glob
from pandas import DataFrame
import pandas as pd
from azureml.logging import get_azureml_logger
from cntk.ops.functions import load_model

base_container = 'data-files'
test_folder = 'test_ctf'
base_local = '/data/bikeshare/'
blob_acct_name = 'kpmgstorage1'
blob_acct_key = '2+BXi305SN45G9yyhykvp7Ij6KYka9W/WvRH4aG5fOuK+9Fenk5Yhg6X6lUMrxjpxE4wKxXyk9NYptzUpZYQkQ=='

final_test_file = base_local + test_folder + "/" + 'sample.ctf'
#Download the train data into our docker container
bbs = BlockBlobService(account_name=blob_acct_name, account_key=blob_acct_key)

def download_and_merge_ctf(remote_base, local_base, relative, output_file_name):
    generator = bbs.list_blobs(remote_base)
    for blob in generator:
        tokens = blob.name.split('/')
        if(tokens[0] == relative
          and len(tokens) > 1
          and tokens[1] != "_SUCCESS"):
            if not os.path.exists(local_base + relative):
                os.makedirs(local_base + relative)
            bbs.get_blob_to_path(remote_base, blob.name, local_base + '/' + blob.name + '.ctf')


    read_files = glob.glob(local_base + relative + "/*.ctf")
    with open(output_file_name, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())


#download_and_merge_ctf(base_container, base_local, test_folder, final_test_file)
with open(final_test_file) as f:
    for line in f:
        print("line len: ", len(line))
        data = line.strip().split(' ')
        print(data)
        target_value = data[1]
        features = data[3:]
        print(features)
print(target_value, len(features))        
#feature_array = np.array(features, dtype=np.float32)
z = load_model('.\outputs\model.dnn')
for index in range(len(z.outputs)):
   print("Index {} for output: {}.".format(index, z.outputs[index].name))



print("args:", z.arguments[0])
#predictions = np.squeeze(z.eval({z.arguments[0]:features}))
