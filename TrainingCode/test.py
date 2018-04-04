from __future__ import print_function
from azure.storage.blob import BlockBlobService
import os
import cntk as C
import numpy as np
import scipy.sparse
import sys
import glob
from azureml.logging import get_azureml_logger