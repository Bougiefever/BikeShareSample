#RUN THIS SECOND

#Create model management account
#az ml account modelmanagement create --location <e.g. eastus2> -n <new model management account name> -g <existing resource group name> --sku-name S1
az ml account modelmanagement create --location eastus2 -n bikesharemanagement -g ml-learning --sku-name S1

#Set model management account
#az ml account modelmanagement set -n <youracctname> -g <yourresourcegroupname>
az ml account modelmanagement set -n bikesharemanagement -g ml-learning
