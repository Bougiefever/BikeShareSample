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

final_test_file = base_local + test_folder + "/" + 'test_complete.ctf'


with open(final_test_file) as f:
  content = f.readlines()

print(content[0])
arr = content[0].strip().split(' ')[3:]
print(arr)

# print("args:", z.arguments[0])
#predictions = np.squeeze(z.eval({z.arguments[0]:features}))
