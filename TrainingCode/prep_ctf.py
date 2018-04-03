import numpy as np
import pyspark
from pyspark.sql.types import Row

spark = pyspark.sql.SparkSession.builder.appName('BikeShare').getOrCreate()

df = spark.read.option("header", "false").csv("wasb://data-files@kpmgstorage1.blob.core.windows.net/traindata/*.csv")

c0_unique = df.select('_c0').distinct().rdd.map(lambda r: r[0]).collect()
c1_unique = df.select('_c1').distinct().rdd.map(lambda r: r[0]).collect()
c5_unique = df.select('_c5').distinct().rdd.map(lambda r: r[0]).collect()
c6_unique = df.select('_c6').distinct().rdd.map(lambda r: r[0]).collect()

#Define Function to make this easier to deal with
def One_Hot_String(val_list, val, sep=' '):
    array = np.eye(len(val_list))[val_list.index(val)].astype(int)
    return sep.join(str(i) for i in array.tolist())

#Apply Function to each column
mapped = df.rdd.map(lambda r: ( Row(One_Hot_String(c0_unique, r['_c0'])), 
                                r['_c1'], 
                                r['_c2'], 
                                r['_c3'],
                                r['_c4'],
                                r['_c5'],
                                r['_c6'])).toDF()

mapped = mapped.rdd.map(lambda r: ( r['_1'], 
                                Row(One_Hot_String(c1_unique, r['_2'])), 
                                r['_3'], 
                                r['_4'],
                                r['_5'],
                                r['_6'],
                                r['_7'])).toDF()

mapped = mapped.rdd.map(lambda r: ( r['_1'], 
                                r['_2'], 
                                r['_3'], 
                                r['_4'],
                                r['_5'],
                                Row(One_Hot_String(c5_unique, r['_6'])),
                                r['_7'])).toDF()

mapped = mapped.rdd.map(lambda r: ( r['_1'], 
                                r['_2'], 
                                r['_3'], 
                                r['_4'],
                                r['_5'],
                                r['_6'],
                                Row(One_Hot_String(c6_unique, r['_7'])) )).toDF()

def row_to_ctf_string(r):
    s = '|label ' + str(r['_7'][0]) + ' '
    s = s + '|features ' + str(r['_1'][0]) + ' '
    s = s + str(r['_2'][0]) + ' '
    s = s + str(r['_3'][0]) + ' '
    s = s + str(r['_4'][0]) + ' '
    s = s + str(r['_5'][0]) + ' '
    s = s + str(r['_6'][0]) + ' '
    return s
pfw = mapped.rdd.flatMap(lambda r: Row(row_to_ctf_string(r)) )
pfw.saveAsTextFile("wasb://data-files@kpmgstorage1.blob.core.windows.net/train_ctf")
