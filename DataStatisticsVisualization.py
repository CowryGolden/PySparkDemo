#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: zhoujincheng\n
@license: (C) Copyright 2018-2020\n
@contact: zhoujincheng777@gmail.com\n
@version: V1.0.0\n
@file: DataStatisticsVisualization.py\n
@time: 2019-10-14 15:10\n
@desc: 数据统计与可视化
'''
import  os, sys
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row


os.environ['SPARK_HOME'] = 'D:\ProgramFiles\spark\spark-2.4.4-hadoop2.6'
os.environ['JAVA_HOME'] = 'D:\ProgramFiles\Java\jdk1.8.0_131'
os.environ['HADOOP_HOME'] = 'D:\ProgramFiles\hadoop\hadoop-2.6.4'
os.environ['PYSPARK_PYTHON'] = 'D:\ProgramFiles\Python364\python.exe'  # 在dos命令行下执行时需要，在IDE中运行配置到Run Configurations中即可，只能使用Python364中的python.exe，而Anaconda3中的python.exe报错：Cannot run program "?D:\ProgramFiles\Anaconda3\python.exe": CreateProcess error=2,

def CreateSparkContext():
    '''
    创建SparkContext对象
    :return:
    '''
    sparkConf = SparkConf().setAppName('DataStatisticsVisualization').set('spark.ui.showConsoleProgress', 'false')
    sc = SparkContext(conf=sparkConf)
    print('master=' + sc.master)
    SetLogger(sc)
    SetPath(sc)
    return sc

def CreateSqlContext():
    '''
    创建SqlContext对象
    :return:
    '''
    sqlContext = SparkSession.builder.getOrCreate()
    return sqlContext

def SetLogger(sc):
    '''
    设置日志级别
    :param sc:
    :return:
    '''
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger('org').setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger('akka').setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def SetPath(sc):
    '''
    设置文件系统路径
    :param sc:
    :return:
    '''
    global Path
    if sc.master[0:5] == 'local':
        Path = u'file:/E:/SparkPythonWorkspace/PySparkDemo/'
    else:
        Path = u'hdfs://master:9000/user/root/'

def PrepareData(sc):
    '''
    准备数据
    :param sc:
    :return: (userRDD, zipcodeRDD)
    '''
    print('======================= 准备数据 =======================')
    # ----------------------------- 1. 导入并转换数据 -----------------------------
    print('========== [PrepareData] >>>> 开始导入 u.user 数据....')
    rawUserData = sc.textFile(Path + u'data/ml-100k/u.user')
    userRDD = rawUserData.map(lambda line: line.split('|'))
    print('========== [PrepareData] >>>> u.user 数据共 ' + str(userRDD.count()) + ' 项')
    print('========== [PrepareData] >>>> 开始导入 free-zipcode-database-Primary.csv 数据....')
    rawZipDataWithHeader = sc.textFile(Path + u'data/ml-100k/free-zipcode-database-Primary.csv')
    zipHeader = rawZipDataWithHeader.first()
    rawZipData = rawZipDataWithHeader.filter(lambda x: x != zipHeader)
    print('========== [PrepareData] >>>> free-zipcode-database-Primary.csv 数据共 ' + str(rawZipData.count()) + ' 项')
    rZipData = rawZipData.map(lambda x: x.replace('\"', ''))
    zipcodeRDD = rZipData.map(lambda x: x.split(','))
    return (userRDD, zipcodeRDD)

def getRowData(userRDD, zipcodeRDD):
    userRows = userRDD.map(lambda r:
                            Row(
                                userid = int(r[0]),
                                age = int(r[1]),
                                gender = r[2],
                                occupation = r[3],
                                zipcode = r[4]
                            )
                           )
    zipcodeRows = zipcodeRDD.map(lambda r:
                                    Row(
                                        zipcode = int(r[0]),
                                        zipCodeType = r[1],
                                        city = r[2],
                                        state = r[3]
                                    )
                                 )
    return (userRows, zipcodeRows)

def getDataFrame(sqlContext, userRows, zipcodeRows):
    userDF = sqlContext.createDataFrame(userRows)
    zipcodeDF = sqlContext.createDataFrame(zipcodeRows)
    print('========== [getDataFrame] >>>> 查看数据 user DataFrame 的 Schema ....')
    userDF.printSchema()
    print('========== [getDataFrame] >>>> 查看数据 zipcode DataFrame 的 Schema ....')
    zipcodeDF.printSchema()
    return (userDF, zipcodeDF)

if __name__ == '__main__':
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ Start DataStatisticsVisualization @@@@@@@@@@@@@@@@@@@@@@@@@')
    sc = CreateSparkContext()
    sqlContext = CreateSqlContext()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ 数据准备阶段 @@@@@@@@@@@@@@@@@@@@@@@@@')
    (userRDD, zipcodeRDD) = PrepareData(sc)
    (userRows, zipcodeRows) = getRowData(userRDD, zipcodeRDD)
    (userDF, zipcodeDF) = getDataFrame(sqlContext, userRows, zipcodeRows)
    # userDF.show(10)
    # zipcodeDF.show(100)
    udf = userDF.alias('udf')  # 创建 DataFrame 别名
    zdf = zipcodeDF.alias('zdf')
    # udf.show(5)
    # zdf.show(50)
    udf.registerTempTable('user_table')  # 注册临时表（传入临时表名称）
    zdf.registerTempTable('zipcode_table')

    # sqlContext.sql('select count(*) counts from user_table').show()  # 使用Spark SQL统计数据项数量
    # sqlContext.sql('select * from user_table').show()  # 使用Spark SQL查看数据内容，默认显示前20条数据
    # sqlContext.sql('select * from user_table limit 5').show()  # 使用Spark SQL查看前5条数据内容
    # udf.select('userid', 'occupation', 'gender', 'age').show(5)  # 使用Spark SQL选取具体字段查看前5条数据内容(方法一)
    # udf.select(udf.userid, udf.occupation, udf.gender, udf.age).show(5)  # 使用Spark SQL选取具体字段查看前5条数据内容(方法二)
    # udf[udf['userid'], udf['occupation'], udf['gender'], udf['age']].show(5)  # 使用Spark SQL选取具体字段查看前5条数据内容(方法三)
    # sqlContext.sql('select userid, occupation, gender, age from user_table').show(5)  # 使用Spark SQL选取具体字段查看前5条数据内容(方法四)
    # sqlContext.sql('select userid, occupation, gender, age, (2019-age) birthyear from user_table').show(5)  # 使用Spark SQL选取已有具体字段并增加计算字段，查看前5条数据内容
    # udf.filter('occupation="technician"').filter('gender="M"').filter('age=24').show()  # 使用Spark SQL按条件筛选记录(方法一)
    # udf.filter((udf.occupation=='technician') & (udf.gender=='M') & (udf.age==24)).show()  # 使用Spark SQL按条件筛选记录(方法二)
    # sqlContext.sql('select * from user_table where occupation="technician" and gender="M" and age=24').show()  # 使用Spark SQL按条件筛选记录(方法三)
    # sqlContext.sql('select userid, occupation, gender, age from user_table order by age').show(5)  # 使用Spark SQL按年龄字段升序排序(方法一)
    # sqlContext.sql('select userid, occupation, gender, age from user_table order by age desc').show(5)  # 使用Spark SQL按年龄字段降序排序(方法二)
    # udf.select('userid', 'occupation', 'gender', 'age').orderBy('age').show(5)  # 使用Spark SQL按年龄字段升序排序(方法三)
    # udf.select('userid', 'occupation', 'gender', 'age').orderBy('age', ascending=0).show(5)  # 使用Spark SQL按年龄字段降序排序(方法四)
    # udf.select(udf.userid, udf.occupation, udf.gender, udf.age).orderBy(udf.age).show(5) # 使用Spark SQL按年龄字段升序排序(方法五)
    # udf.select(udf.userid, udf.occupation, udf.gender, udf.age).orderBy(udf.age.desc()).show(5)  # 使用Spark SQL按年龄字段降序排序(方法六)
    # sqlContext.sql('select userid, occupation, age, gender, zipcode from user_table order by age desc, gender').show(5)  # 使用Spark SQL按多个字段排序(方法一)
    # udf.select('userid', 'occupation', 'age', 'gender', 'zipcode').orderBy(['age', 'gender'], ascending=[0, 1]).show(5)  # 使用Spark SQL按多个字段排序(方法二)
    # udf.select(udf.userid, udf.occupation, udf.age, udf.gender, udf.zipcode).orderBy(udf.age.desc(), udf.gender).show(5)  # 使用Spark SQL按多个字段排序(方法三)
    # sqlContext.sql('select distinct gender from user_table').show()  # 使用Spark SQL按性别去重
    # udf.select('gender').distinct().show()
    # sqlContext.sql('select distinct age, gender from user_table').show()  # 使用Spark SQL按年龄+性别去重
    # udf.select('age', 'gender').distinct().show()
    # sqlContext.sql('select gender, count(*) counts from user_table group by gender').show()  # 使用Spark SQL按性别分组统计
    # udf.select('gender').groupBy('gender').count().show()
    # sqlContext.sql('select gender, occupation, count(*) counts from user_table group by gender, occupation').show(10)  # 使用Spark SQL按性别+职业分组统计
    # udf.select('gender', 'occupation').groupBy('gender', 'occupation').count().orderBy('gender', 'occupation').show(10)
    # udf.stat.crosstab('occupation', 'gender').show(30)  # 交叉统计表
    # sqlContext.sql('select u.*, z.city, z.state from user_table u left join zipcode_table z on u.zipcode = z.zipcode where z.state = "NY"').show(10)  # 多表连接查询
    # sqlContext.sql('select z.state, count(*) counts from user_table u left join zipcode_table z on u.zipcode = z.zipcode group by z.state').show(60)  # 多表连接查询
    joindf = udf.join(zdf, udf.zipcode == zdf.zipcode, 'left_outer')  # 多表连接
    joindf.printSchema()
    # joindf.groupBy('state').count().show(60)

    # ---------------------------- 将Spark DataFrames 转换为 Pandas DataFrames，并绘制图形 -------------------------------
    # groupByState_df = joindf.groupBy('state').count()
    # groupByState_pandas_df = groupByState_df.toPandas().set_index('state')  # 使用toPandas()将groupByState_df转换为Pandas DataFrames，并将州字段state设置为索引
    # ax = groupByState_pandas_df['count'].plot(kind='bar', title='State', figsize=(12, 6), legend=True, fontsize=12)  # 用Pandas DataFrames（配合matplotlib模块）将数据绘出直方图Bar Chart
    # plt.show()

    occupation_df = sqlContext.sql('select u.occupation, count(*) as counts from user_table u group by u.occupation')  # 按不同职业统计人数并绘制饼形图
    occupation_pandas_df = occupation_df.toPandas().set_index('occupation')  # 转换为Pandas DataFrames，并将职业字段occupation设置为索引
    ax = occupation_pandas_df['counts'].plot(kind='pie', title='occupation', figsize=(10, 10), startangle=90, autopct='%1.1f%%')  # 用Pandas DataFrames（配合matplotlib模块）将数据绘出饼形图Pie Chart
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)  # 图例显示在右边
    plt.show()

