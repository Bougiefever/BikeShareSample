#RUN THIS THIRD

#Set ENV
az ml env set -g dacrookbikesharerg -n dacrookbikeshare

#Create Real Time Web Service
az ml service create realtime -f score.py --model-file model.pkl -s service_schema.json -n irisapp -r python --collect-model-data true
