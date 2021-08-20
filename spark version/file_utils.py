from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as fn
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf


spark = SparkSession.builder.getOrCreate()
DATASET_NAME='data 2.91G 9000000.csv'
global call
call=1
def ToInt(x):
    ret=[]
    for i in range(0,len(x)):
        if x[i]=='True':
            ret[i]=1
        elif x[i]=='False':
            ret[i]=0
        else:
            ret[i]=int(x[i])
    return ret
def read_data(begin_row,end_row):
    properties={}
    origin=spark.read.csv("DATASET_NAME")
    origin.registerTempTable("origin")
    main_data=spark.sql("select _c4,_c5,_c6,_c7,_c8,_c9,_c10,_c12+_c19+_c26+_c33+_c40,_c13+_c20+_c27+_c34+_c41,_c14+_c21+_c28+_c35+_c42,_c15+_c22+_c29+_c36+_c43 from origin where _c0>="+str(begin_row)+" and _c0<"+str(end_row))
    properties=main_data.toDF('class','team1_firstBlood','team1_firstTower','team1_firstInhibitor','team1_firstBaron','team1_firstDragon','team1_firstRiftHerald','player_kills','player_deaths','player_assists','player_goldEarned')
    properties=properties.withColumn('player_kills',properties['player_kills'].cast(IntegerType()))
    properties=properties.withColumn('player_deaths',properties['player_deaths'].cast(IntegerType()))
    properties=properties.withColumn('player_assists',properties['player_assists'].cast(IntegerType()))
    properties=properties.withColumn('player_goldEarned',properties['player_goldEarned'].cast(IntegerType()))
    types=np.array([0,0,0,0,0,0,1,1,1,1])
    return types,properties
def generate_cross_verify(train,test):
    global call
    origin=spark.read.csv(DATASET_NAME)
    #origin=origin.filter(origin._c0 < 100)
    origin=origin.filter(origin._c0 != 'index')
    train,test=origin.randomSplit([train/(train+test),test/(train+test)],100*call)
    train.registerTempTable("train")
    test.registerTempTable("test")
    train=spark.sql("select _c4,_c5,_c6,_c7,_c8,_c9,_c10,_c12+_c19+_c26+_c33+_c40,_c13+_c20+_c27+_c34+_c41,_c14+_c21+_c28+_c35+_c42,_c15+_c22+_c29+_c36+_c43 from train")
    test=spark.sql("select _c4,_c5,_c6,_c7,_c8,_c9,_c10,_c12+_c19+_c26+_c33+_c40,_c13+_c20+_c27+_c34+_c41,_c14+_c21+_c28+_c35+_c42,_c15+_c22+_c29+_c36+_c43 from test")
    train=train.toDF('class','team1_firstBlood','team1_firstTower','team1_firstInhibitor','team1_firstBaron','team1_firstDragon','team1_firstRiftHerald','player_kills','player_deaths','player_assists','player_goldEarned')
    test=test.toDF('class','team1_firstBlood','team1_firstTower','team1_firstInhibitor','team1_firstBaron','team1_firstDragon','team1_firstRiftHerald','player_kills','player_deaths','player_assists','player_goldEarned')
    train=train.withColumn('player_kills',train['player_kills'].cast(IntegerType()))
    train=train.withColumn('player_deaths',train['player_deaths'].cast(IntegerType()))
    train=train.withColumn('player_assists',train['player_assists'].cast(IntegerType()))
    train=train.withColumn('player_goldEarned',train['player_goldEarned'].cast(IntegerType()))
    test=test.withColumn('player_kills',test['player_kills'].cast(IntegerType()))
    test=test.withColumn('player_deaths',test['player_deaths'].cast(IntegerType()))
    test=test.withColumn('player_assists',test['player_assists'].cast(IntegerType()))
    test=test.withColumn('player_goldEarned',test['player_goldEarned'].cast(IntegerType()))
    train.persist()
    test.persist()
    types=np.array([0,0,0,0,0,0,1,1,1,1])
    call+=1
    return types,train,test
if __name__=='__main__':
    for i in range(0,5):
        _,train,test=generate_cross_verify(4,1)
        train.show(5)

