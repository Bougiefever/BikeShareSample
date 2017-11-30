#RUN THIS FIRST

#Setup if you plan to deploy to Azure
#az ml env setup -n <new deployment environment name> --location <e.g. eastus2>
az ml env setup -n dacrookbikeshare -l eastus2

#Show progress
az ml env show -g dacrookbikesharerg -n dacrookbikeshare