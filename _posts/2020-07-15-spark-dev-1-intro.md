---
title: "Spark Dev - 1/5 a gentle introduction"
date: 2020-07-15
categories:
  - Data Engineering / Spark
tags: [Data Engineering / Spark]
header:
  image: "/images/banners/spark-dev.jpg"
excerpt: "How to create a session, select and manipulate columns, retrieve basic infos & more"
mathjax: "true"
---

Banner from a cropped photo by Peter Spencer from Pexels

In order to prepare the Databricks' Associate Developer for Apache Spark 2.4 certification, i've made all the examples of the book ["Spark: The Definitive Guide" 
by Bill Chambers, Matei Zaharia (O'reilly - Feb 2018](https://www.oreilly.com/library/view/spark-the-definitive/9781491912201/) as exercices. This book is an invaluable resource ! There are from time to time several variations from the orginal examples. 

This blog post is part of a serie about Spark Dev :
- 01 introduction
- [02 basic structured_ops]()
- [03 aggregations]()
- [04 working with different types of data]()
- [05 joins]()

You can find all these jupyter notebooks in a dedicated github repository, with for each a blank notebook (without code / only result) in order to train yourselves. 

Create a spark session


```scala
import org.apache.spark.sql.SparkSession
```


    Intitializing Scala interpreter ...



    Spark Web UI available at http://ed1efe135804:4040
    SparkContext available as 'sc' (version = 3.0.0, master = local[*], app id = local-1597170050923)
    SparkSession available as 'spark'






    import org.apache.spark.sql.SparkSession





```scala
val spark = SparkSession.builder
    .appName("essdg")
    .getOrCreate()
```




    spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@4a9b34fb





```scala
!ls ../../../src
```

    01.gentle_intro_COMPLEMENT.ipynb
    01.gentle_intro.ipynb
    01.gentle_intro_TEST.ipynb
    02.basic_structured_op_COMPLEMENT.ipynb
    02.basic_structured_op.ipynb
    02.basic_structured_op_TEST.ipynb
    03.aggregations_COMPLETE.ipynb
    03.aggregations_INCOMPLETE.ipynb
    03.aggregations.ipynb
    04.working_with_different_types_of_data.ipynb
    05.joins.ipynb
    05.joins_TEST.ipynb
    201508_station_data.csv
    201508_trip_data.csv
    spark_docker.txt
    


Create a DF with range


```scala
var test_df = spark.range(100).toDF("numb")
test_df.show(5)
```

    +----+
    |numb|
    +----+
    |   0|
    |   1|
    |   2|
    |   3|
    |   4|
    +----+
    only showing top 5 rows
    





    test_df: org.apache.spark.sql.DataFrame = [numb: bigint]




Retrieve only number divisible by 2


```scala
test_df.where("numb % 2 = 0").show(5)
```

    +----+
    |numb|
    +----+
    |   0|
    |   2|
    |   4|
    |   6|
    |   8|
    +----+
    only showing top 5 rows
    


Read a csv and load it into a DF


```scala
val df = spark.read
    .format("csv")
    .option("header", "True")
    .option("inferSchema", "True")
    .load("../../../src/201508_trip_data.csv")

df.select("Trip ID", "Duration", "Start Date", "Start Station", "Start Terminal", "End Date").show(5)
```

    +-------+--------+---------------+--------------------+--------------+---------------+
    |Trip ID|Duration|     Start Date|       Start Station|Start Terminal|       End Date|
    +-------+--------+---------------+--------------------+--------------+---------------+
    | 913460|     765|8/31/2015 23:26|Harry Bridges Pla...|            50|8/31/2015 23:39|
    | 913459|    1036|8/31/2015 23:11|San Antonio Shopp...|            31|8/31/2015 23:28|
    | 913455|     307|8/31/2015 23:13|      Post at Kearny|            47|8/31/2015 23:18|
    | 913454|     409|8/31/2015 23:10|  San Jose City Hall|            10|8/31/2015 23:17|
    | 913453|     789|8/31/2015 23:09|Embarcadero at Fo...|            51|8/31/2015 23:22|
    +-------+--------+---------------+--------------------+--------------+---------------+
    only showing top 5 rows
    





    df: org.apache.spark.sql.DataFrame = [Trip ID: int, Duration: int ... 9 more fields]




You can make any DataFrame into a table or view with one simple method call. This will allow us to query this table like an SQL one :


```scala
df.createOrReplaceTempView("df")
import org.apache.spark.sql.functions._
```




    import org.apache.spark.sql.functions._




Select the first 2 rows


```scala
df.columns
```




    res4: Array[String] = Array(Trip ID, Duration, Start Date, Start Station, Start Terminal, End Date, End Station, End Terminal, Bike #, Subscriber Type, Zip Code)





```scala
df.select("Trip ID", "Duration", "Start Date", "Start Station").show(2) // "*" to select all cols
```

    +-------+--------+---------------+--------------------+
    |Trip ID|Duration|     Start Date|       Start Station|
    +-------+--------+---------------+--------------------+
    | 913460|     765|8/31/2015 23:26|Harry Bridges Pla...|
    | 913459|    1036|8/31/2015 23:11|San Antonio Shopp...|
    +-------+--------+---------------+--------------------+
    only showing top 2 rows
    


Retrieve the first 3 lines as an array


```scala
df.take(3)
```




    res6: Array[org.apache.spark.sql.Row] = Array([913460,765,8/31/2015 23:26,Harry Bridges Plaza (Ferry Building),50,8/31/2015 23:39,San Francisco Caltrain (Townsend at 4th),70,288,Subscriber,2139], [913459,1036,8/31/2015 23:11,San Antonio Shopping Center,31,8/31/2015 23:28,Mountain View City Hall,27,35,Subscriber,95032], [913455,307,8/31/2015 23:13,Post at Kearny,47,8/31/2015 23:18,2nd at South Park,64,468,Subscriber,94107])




Get Schema of the DF


```scala
df.schema
```




    res7: org.apache.spark.sql.types.StructType = StructType(StructField(Trip ID,IntegerType,true), StructField(Duration,IntegerType,true), StructField(Start Date,StringType,true), StructField(Start Station,StringType,true), StructField(Start Terminal,IntegerType,true), StructField(End Date,StringType,true), StructField(End Station,StringType,true), StructField(End Terminal,IntegerType,true), StructField(Bike #,IntegerType,true), StructField(Subscriber Type,StringType,true), StructField(Zip Code,StringType,true))




Print it more nicely


```scala
df.printSchema()
```

    root
     |-- Trip ID: integer (nullable = true)
     |-- Duration: integer (nullable = true)
     |-- Start Date: string (nullable = true)
     |-- Start Station: string (nullable = true)
     |-- Start Terminal: integer (nullable = true)
     |-- End Date: string (nullable = true)
     |-- End Station: string (nullable = true)
     |-- End Terminal: integer (nullable = true)
     |-- Bike #: integer (nullable = true)
     |-- Subscriber Type: string (nullable = true)
     |-- Zip Code: string (nullable = true)
    


Sort the DF by the Duration col


```scala
spark.sql("""
SELECT * from df
ORDER Duration ASC
""").show(4)

df.sort(asc("Duration")).show(5)
```

    +-------+--------+----------------+--------------------+--------------+----------------+--------------------+------------+------+---------------+--------+
    |Trip ID|Duration|      Start Date|       Start Station|Start Terminal|        End Date|         End Station|End Terminal|Bike #|Subscriber Type|Zip Code|
    +-------+--------+----------------+--------------------+--------------+----------------+--------------------+------------+------+---------------+--------+
    | 508274|      60|10/21/2014 11:57|San Francisco Cal...|            69|10/21/2014 11:58|San Francisco Cal...|          69|   578|     Subscriber|   94107|
    | 506025|      60| 10/20/2014 8:16|   Market at Sansome|            77| 10/20/2014 8:17|   Market at Sansome|          77|   109|     Subscriber|   94114|
    | 483333|      60| 10/4/2014 19:21|Yerba Buena Cente...|            68| 10/4/2014 19:22|Yerba Buena Cente...|          68|   560|       Customer|     nil|
    | 473451|      60|  9/29/2014 7:38|Civic Center BART...|            72|  9/29/2014 7:39|Civic Center BART...|          72|   358|     Subscriber|   94062|
    | 438041|      60|  9/4/2014 10:53|Civic Center BART...|            72|  9/4/2014 10:54|Civic Center BART...|          72|   291|     Subscriber|   94117|
    +-------+--------+----------------+--------------------+--------------+----------------+--------------------+------------+------+---------------+--------+
    only showing top 5 rows
    


In ascending number & print the physical plan


```scala
df.sort(asc("Duration")).explain
```

    == Physical Plan ==
    *(1) Sort [Duration#31 ASC NULLS FIRST], true, 0
    +- Exchange rangepartitioning(Duration#31 ASC NULLS FIRST, 200), true, [id=#87]
       +- FileScan csv [Trip ID#30,Duration#31,Start Date#32,Start Station#33,Start Terminal#34,End Date#35,End Station#36,End Terminal#37,Bike ##38,Subscriber Type#39,Zip Code#40] Batched: false, DataFilters: [], Format: CSV, Location: InMemoryFileIndex[file:/src/201508_trip_data.csv], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<Trip ID:int,Duration:int,Start Date:string,Start Station:string,Start Terminal:int,End Dat...
    
    


Query the table to display cols in an ordered way :


```scala
spark.sql("""
SELECT Duration, `Start Date`, `End Station` FROM df
ORDER BY Duration DESC
LIMIT 5
""").show()
```

    +--------+---------------+--------------------+
    |Duration|     Start Date|         End Station|
    +--------+---------------+--------------------+
    |17270400|12/6/2014 21:59|       2nd at Folsom|
    | 2137000|6/28/2015 21:50|Yerba Buena Cente...|
    | 1852590|  5/2/2015 6:17|Castro Street and...|
    | 1133540|7/10/2015 10:35|University and Em...|
    |  720454|10/30/2014 8:29|Stanford in Redwo...|
    +--------+---------------+--------------------+
    


Count the "end station" col when groupped


```scala
spark.sql("""
SELECT `End Station`, count(`End Station`) 
FROM df
GROUP BY `End Station`
LIMIT 5
""").show()
```

    +--------------------+------------------+
    |         End Station|count(End Station)|
    +--------------------+------------------+
    |       2nd at Folsom|              4727|
    |California Ave Ca...|               496|
    |Powell at Post (U...|              4134|
    | Golden Gate at Polk|              2852|
    |Yerba Buena Cente...|              6288|
    +--------------------+------------------+
    


Same thing in a scala way


```scala
df.groupBy("End Station").count().show(5)
```

    +--------------------+-----+
    |         End Station|count|
    +--------------------+-----+
    |       2nd at Folsom| 4727|
    |California Ave Ca...|  496|
    |Powell at Post (U...| 4134|
    | Golden Gate at Polk| 2852|
    |Yerba Buena Cente...| 6288|
    +--------------------+-----+
    only showing top 5 rows
    



```scala
df.groupBy("End Station").count().orderBy(desc("count")).show(4)

spark.sql("""
select `End Station`, count(`End Station`) as count
from df
group by `End Station`
order by count DESC
""").show(3)
```

    +--------------------+-----+
    |         End Station|count|
    +--------------------+-----+
    |San Francisco Cal...|34810|
    |San Francisco Cal...|22523|
    |Harry Bridges Pla...|17810|
    +--------------------+-----+
    only showing top 3 rows
    



```scala
df.selectExpr("count(Duration)").show(4)
```

    +---------------+
    |count(Duration)|
    +---------------+
    |         354152|
    +---------------+
    



```scala
df.selectExpr("mean(Duration)", "avg(Duration)").show(4)
```

    +------------------+------------------+
    |    mean(Duration)|     avg(Duration)|
    +------------------+------------------+
    |1046.0326611172604|1046.0326611172604|
    +------------------+------------------+
    


Retrieve min and max of a col


```scala
df.select(min("Duration"), max("Duration")).show()

spark.sql("""
SELECT min(Duration), max(Duration)
FROM df
""").show()
```

    +-------------+-------------+
    |min(Duration)|max(Duration)|
    +-------------+-------------+
    |           60|     17270400|
    +-------------+-------------+
    



```scala
df.groupBy("End Station").agg(min("Duration"), max("Duration")).show(5)

spark.sql("""
SELECT `End Station`, min(Duration), max(Duration)
FROM df
GROUP BY `End Station`
""").show(5)
```

    +--------------------+-------------+-------------+
    |         End Station|min(Duration)|max(Duration)|
    +--------------------+-------------+-------------+
    |       2nd at Folsom|           61|     17270400|
    |California Ave Ca...|           82|       688899|
    |Powell at Post (U...|           66|       141039|
    | Golden Gate at Polk|           60|       238286|
    |Yerba Buena Cente...|           60|      2137000|
    +--------------------+-------------+-------------+
    only showing top 5 rows
    


and in scala:


```scala
df.groupBy("End Station").sum("Duration").withColumnRenamed("sum(Duration)", "SUM_DURATION").show(4)

spark.sql("""
select `End Station`, sum(Duration) as SUM_DURATION
from df
group by `End Station`
""").show(4)
```

    +--------------------+------------+
    |         End Station|SUM_DURATION|
    +--------------------+------------+
    |       2nd at Folsom|    21031718|
    |California Ave Ca...|     2629339|
    |Powell at Post (U...|     8691192|
    | Golden Gate at Polk|     4531730|
    +--------------------+------------+
    only showing top 4 rows
    



```scala
df.groupBy("End Station").min("Duration").sort(asc("min(Duration)")).show(30)
```

    +--------------------+-------------+
    |         End Station|min(Duration)|
    +--------------------+-------------+
    |San Francisco Cal...|           60|
    |       Howard at 2nd|           60|
    |   Market at Sansome|           60|
    |   2nd at South Park|           60|
    |Yerba Buena Cente...|           60|
    |Embarcadero at Fo...|           60|
    |Embarcadero at Sa...|           60|
    |     2nd at Townsend|           60|
    |  Powell Street BART|           60|
    |     Beale at Market|           60|
    | Golden Gate at Polk|           60|
    |   Steuart at Market|           60|
    |      Market at 10th|           60|
    |Harry Bridges Pla...|           60|
    |     Spear at Folsom|           60|
    |Temporary Transba...|           60|
    |Civic Center BART...|           60|
    |San Francisco Cal...|           60|
    |Embarcadero at Va...|           61|
    |       2nd at Folsom|           61|
    |       Market at 4th|           61|
    |     Townsend at 7th|           61|
    |Mechanics Plaza (...|           61|
    |San Antonio Caltr...|           61|
    |Washington at Kearny|           61|
    |Embarcadero at Br...|           61|
    |San Jose Diridon ...|           62|
    |South Van Ness at...|           62|
    |Commercial at Mon...|           62|
    |San Antonio Shopp...|           62|
    +--------------------+-------------+
    only showing top 30 rows
    



```scala
df.groupBy("End Station").agg(min("Duration"), max("Duration")).show(5)
```

    +--------------------+-------------+-------------+
    |         End Station|min(Duration)|max(Duration)|
    +--------------------+-------------+-------------+
    |       2nd at Folsom|           61|     17270400|
    |California Ave Ca...|           82|       688899|
    |Powell at Post (U...|           66|       141039|
    | Golden Gate at Polk|           60|       238286|
    |Yerba Buena Cente...|           60|      2137000|
    +--------------------+-------------+-------------+
    only showing top 5 rows
    



```scala
df.groupBy("End Station").sum("Duration").withColumnRenamed("sum(Duration)", "SUM_DURATION").show(5)
```

    +--------------------+------------+
    |         End Station|SUM_DURATION|
    +--------------------+------------+
    |       2nd at Folsom|    21031718|
    |California Ave Ca...|     2629339|
    |Powell at Post (U...|     8691192|
    | Golden Gate at Polk|     4531730|
    |Yerba Buena Cente...|     6658500|
    +--------------------+------------+
    only showing top 5 rows
    


rename a col


```scala
val df_renamed = df.withColumnRenamed("End Station", "End_station")
df_renamed.select("Duration", "End_station", "Bike #").show(2)
```

    +--------+--------------------+------+
    |Duration|         End_station|Bike #|
    +--------+--------------------+------+
    |     765|San Francisco Cal...|   288|
    |    1036|Mountain View Cit...|    35|
    +--------+--------------------+------+
    only showing top 2 rows
    





    df_renamed: org.apache.spark.sql.DataFrame = [Trip ID: int, Duration: int ... 9 more fields]





```scala
df.select("Duration", "End station", "Bike #").withColumnRenamed("Bike #", "Bike NB").show(4)
```

    +--------+--------------------+-------+
    |Duration|         End station|Bike NB|
    +--------+--------------------+-------+
    |     765|San Francisco Cal...|    288|
    |    1036|Mountain View Cit...|     35|
    |     307|   2nd at South Park|    468|
    |     409| San Salvador at 1st|     68|
    +--------+--------------------+-------+
    only showing top 4 rows
    



```scala
df.selectExpr("Duration as DURATION", "`End station` as `END STATION`").show(4)
```

    +--------+--------------------+
    |DURATION|         END STATION|
    +--------+--------------------+
    |     765|San Francisco Cal...|
    |    1036|Mountain View Cit...|
    |     307|   2nd at South Park|
    |     409| San Salvador at 1st|
    +--------+--------------------+
    only showing top 4 rows
    


That's all for this gentle introduction : hope you've enjoyed it :)
