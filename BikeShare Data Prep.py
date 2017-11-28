import pyspark

from azureml.dataprep.package import run
from pyspark.sql.functions import *

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('BikeShare').getOrCreate()

# dataflow_idx=2 sets the dataflow to the 3rd dataflow (the index starts at 0), the Join Result.
df = run('BikeShare Data Prep.dprep', dataflow_idx=2)
df.show(n=10)
row_count_first = df.count()

# Example file name: 'wasb://data-files@bikesharestorage.blob.core.windows.net/testdata'
blobfolder = 'wasb://data-files@bikesharestorage.blob.core.windows.net/testdata'

df.write.csv(blobfolder, mode='overwrite') 

# retrieve csv file parts into one data frame
csvfiles = "wasb://data-files@bikesharestorage.blob.core.windows.net/testdata/*.csv"
df = spark.read.option("header", "false").csv(csvfiles)
row_count_result = df.count()
print(row_count_result)
if (row_count_first == row_count_result):
    print('counts match')
else:
    print('counts do not match')
print('done')