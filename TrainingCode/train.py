from __future__ import print_function
from azure.storage.blob import BlockBlobService
import os
import cntk as C
import numpy as np
import scipy.sparse
import sys
import glob
from azureml.logging import get_azureml_logger

run_logger = get_azureml_logger()

#minibatch size is size of single minibatch.  To calc optimal size:
#mem = memory - neural net params * float32 size
#size = mem / (num_features * float32 size)
minibatch_size = 10000
#epoch size is virtual concept of whole data set.
epoch_size = 50000
num_features = 212

#sweep is a full pass through the data set.
max_sweeps = 1000

num_outputs = 58
# Define the task.
x = C.input_variable(num_features)
y = C.input_variable(num_outputs, is_sparse=True)
base_container = 'data-files'
train_folder = 'train_ctf'
test_folder = 'test_ctf'
base_local = 'C:/data/bikeshare/'
model_path = os.path.join('.', 'outputs', 'model.dnn')
blob_acct_name = 'bikesharestorage'
blob_acct_key = 'L4XWmq7V6+Xk90GloM7zxNOkp4ut9kzyf7+1e1Qp9Bt+hGiP33jZuT4L9U8Gye4UWlrloXTtOMO+t4J7/3sMiA=='

final_train_file = base_local + train_folder + "/" + 'train_complete.ctf'
final_test_file = base_local + test_folder + "/" + 'test_complete.ctf'

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

download_and_merge_ctf(base_container, base_local, train_folder, final_train_file)
#download_and_merge_ctf(base_container, base_local, test_folder, 'test_complete.ctf')

# Read a COMPOSITE reader to read data from both the image map and CTF files
def create_minibatch_source(ctf_file, is_training, num_outputs):
    
    # create CTF DESERIALIZER for CTF file
    data_source = C.io.CTFDeserializer(ctf_file, C.io.StreamDefs(
        features = C.io.StreamDef(field="features", shape=num_features),
        label = C.io.StreamDef(field="label", shape=num_outputs)))

    # create a minibatch source by compositing them together 
    return C.io.MinibatchSource([data_source], max_sweeps=max_sweeps, randomize=is_training)

train_source = create_minibatch_source(final_train_file, True, num_outputs)
#test_source = create_minibatch_source(final_test_file, False, num_outputs)

input_map = {    
    x : train_source.streams.features,
    y : train_source.streams.label
}

num_neurons = 100
def create_model(x):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.relu):            
            h = C.layers.Dense(num_neurons, name="first")(x)
            h = C.layers.Dense(num_neurons, name="second")(h)
            h = C.layers.Dense(num_neurons, name="third")(h)
            h = C.layers.Dropout(dropout_rate=0.5)(h)
            p = C.layers.Dense(num_outputs, activation = None, name="prediction")(h)         
            return p

model = create_model(x)

# loss function
loss = C.cross_entropy_with_softmax(model, y)

#error metric function
error = C.classification_error(model, y)

#combine error and loss for training usage (AKA Criterion function)
criterion = C.combine([loss, error])

lr_schedule = [0.01]*100 + [0.001]*100 + [0.0001]
learner = C.sgd(model.parameters, lr_schedule, epoch_size=epoch_size)

#frequency is defined by number of samples
progress_writer = C.logging.ProgressPrinter(freq=epoch_size)

#frequence is by number of samples
checkpoint_config = C.CheckpointConfig(filename=model_path, frequency=(epoch_size * 10), restore = False)

#test_config = C.TestConfig(test_source)

progress = criterion.train(train_source, 
                           minibatch_size = minibatch_size,
                           model_inputs_to_streams = input_map,
                           epoch_size = epoch_size,
                           max_epochs = 100, 
                           parameter_learners = [learner], 
                           callbacks = [progress_writer])#, checkpoint_config])#, test_config])

inf_model = C.softmax(model)
inf_model.save(model_path)
