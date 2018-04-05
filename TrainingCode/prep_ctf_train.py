import numpy as np
import pyspark
from pyspark.sql.types import Row

spark = pyspark.sql.SparkSession.builder.appName('BikeShare').getOrCreate()

df = spark.read.option("header", "false").csv("wasb://data-files@kpmgstorage1.blob.core.windows.net/traindata/*.csv")

c0_unique = df.select('_c0').distinct().rdd.map(lambda r: r[0]).collect()
c1_unique = df.select('_c1').distinct().rdd.map(lambda r: r[0]).collect()
#c5_unique = df.select('_c5').distinct().rdd.map(lambda r: r[0]).collect()
c5_unique = ['67.0', '36.0', '46.0', '10.0', '47.0', '107.0', '58.0', '9.0', '88.0', '89.0', '133.0', '27.0', '80.0', '22.0', '39.0', '178.0', '42.0', '4.0', '115.0', '161.0', '102.0', '33.0', '84.0', '25.0', '131.0', '152.0', '51.0', '21.0', '23.0', '32.0', '94.0', '49.0', '54.0', '146.0', '179.0', '30.0', '96.0', '190.0', '8.0', '41.0', '110.0', '197.0', '93.0', '68.0', '145.0', '19.0', '105.0', '180.0', '87.0', '185.0', '119.0', '59.0', '73.0', '118.0', '77.0', '215.0', '16.0', '81.0', '11.0', '213.0', '98.0', '14.0', '195.0', '163.0', '104.0', '78.0', '176.0', '184.0', '76.0', '95.0', '43.0', '74.0', '6.0', '91.0', '169.0', '183.0', '100.0', '116.0', '124.0', '70.0', '7.0', '75.0', '173.0', '151.0', '15.0', '5.0', '139.0', '218.0', '150.0', '31.0', '24.0', '20.0', '3.0', '44.0', '130.0', '17.0', '56.0', '174.0', '142.0', '159.0', '71.0', '40.0', '177.0', '175.0', '160.0', '189.0', '63.0', '108.0', '141.0', '149.0', '137.0', '109.0', '37.0', '117.0', '90.0', '57.0', '121.0', '140.0', '208.0', '72.0', '143.0', '186.0', '85.0', '12.0', '217.0', '200.0', '126.0', '138.0', '170.0', '65.0', '97.0', '210.0', '135.0', '201.0', '171.0', '92.0', '219.0', '29.0', '136.0', '205.0', '196.0', '212.0', '207.0', '1.0', '70.0', 'ERROR.0', '55.0', '125.0', '60.0', '45.0', '53.0', '66.0', '39.0', '109.0', '38.0', '112.0', '42.0', '13.0', '99.0', '129.0', '114.0', '86.0', '48.0', '26.0', '122.0', '111.0', '123.0', '130.0', '64.0', '139.0', '103.0', '113.0', '131.0', '128.0', '120.0', '35.0', '69.0', '106.0', '61.0', '134.0', '50.0', '52.0', '82.0', '79.0', '132.0', '92.0', '162.0', '167.0', '192.0', '193.0', '194.0', '60.0', '52.0', '216.0', '209.0', '211.0', '214.0', '204.0', '202.0', '203.0', '199.0', '153.0']

print(len(c5_unique))
print(len(c0_unique)+len(c1_unique)+len(c5_unique))

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


def row_to_ctf_string(r):
    s = '|label ' +str(r['_7'][0]) + ' ' \
        + '|features ' + str(r['_1'][0]) + ' ' \
        + str(r['_2'][0]) + ' ' \
        + str(r['_3']) + ' ' \
        + str(r['_4']) + ' ' \
        + str(r['_5']) + ' ' \
        + str(r['_6'][0]) + ' '
    return s

pfw = mapped.rdd.flatMap(lambda r: Row(row_to_ctf_string(r)) )

pfw.saveAsTextFile("wasb://data-files@kpmgstorage1.blob.core.windows.net/train_ctf")

