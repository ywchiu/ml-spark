
object Cells {
  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}
  val rawData = sc.textFile("/home/david/spark/test.trans")
  rawData.first()

  /* ... new cell ... */

  val rawRatings = rawData.map(line => line.split(" ")).collect().toSeq
  val rdd = sc.parallelize(rawRatings, 2).cache()
  val fpg = new FPGrowth()
  
      val model = fpg
        .setMinSupport(0.1)
        .setNumPartitions(2)
        .run(rdd)

  /* ... new cell ... */

      val freqItemsets3 = model.freqItemsets.collect().map { itemset =>      
        if(itemset.items.size >= 2){
          println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
        }
      }
}
              