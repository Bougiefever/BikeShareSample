import pyspark

print('Starting')
spark = pyspark.sql.SparkSession.builder.appName('BikeShare').getOrCreate()
# retrieve csv file parts into one data frame

df = spark.read.option("header", "false").csv("wasb://data-files@bikesharestorage.blob.core.windows.net/traindata/*.csv")
df.show(n=10)
num_rows = df.count()
print(num_rows)
print('Complete')
