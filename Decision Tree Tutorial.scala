
object Cells {
  //從檔案讀取資料
  val rawData = sc.textFile("/home/david/churnTrain.csv")

  /* ... new cell ... */

  //去除標頭
  val noheader = rawData.mapPartitionsWithIndex((idx, lines) => {
    if (idx == 0) {
      lines.drop(1)
    }
    lines
  })

  /* ... new cell ... */

  //依逗號切開資料
  val splitlines = noheader.map(lines => {
    lines.split(',')
  })

  /* ... new cell ... */

  import org.apache.spark.mllib.regression.LabeledPoint
  import org.apache.spark.mllib.linalg.Vectors
  val trainData = splitlines.map { col =>      
    val churn = col(col.size - 1)
    val intenational = if (col(4) == "\"no\"") 0.toDouble else 1.toDouble
    val voice = if (col(5) == "\"no\"") 0.toDouble else 1.toDouble
    val label = if (churn == "\"no\"") 0.toInt else 1.toInt
    val features = Array(intenational, voice) ++ col.slice(6, col.size - 1).map(_.toDouble)
    LabeledPoint(label, Vectors.dense(features))
  }
  trainData.first()

  /* ... new cell ... */

  import org.apache.spark.mllib.tree.DecisionTree
  import org.apache.spark.mllib.tree.configuration.Algo
  import org.apache.spark.mllib.tree.impurity.Entropy
  val maxTreeDepth = 5
  val dtModel = DecisionTree.train(trainData, Algo.Classification, Entropy,maxTreeDepth)

  /* ... new cell ... */

  val dataPoint = trainData.first
  val prediction = dtModel.predict(dataPoint.features)

  /* ... new cell ... */

  val dtTotalCorrect = trainData.map { point =>
  if (dtModel.predict(point.features) == point.label) 1 else 0
  }.sum
  val dtAccuracy = dtTotalCorrect / trainData.count

  /* ... new cell ... */

  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  val dtMetrics = Seq(dtModel).map{ model =>
    val scoreAndLabels = trainData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
  }
  
  "For %s: Area Under PR: %f, Area Under AUC %f".format(dtMetrics(0)._1, dtMetrics(0)._2, dtMetrics(0)._3)

  /* ... new cell ... */

  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.tree.impurity.Impurity
  import org.apache.spark.mllib.tree.impurity.Entropy
  import org.apache.spark.mllib.tree.impurity.Gini
  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  /* ... new cell ... */

  val dtResultsEntropy = Seq(1,2,3,4,5,10,20).map { param =>
    val model = trainDTWithParams(trainData, param, Entropy)
    val scoreAndLabels = trainData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param tree depth", metrics.areaUnderROC)
  }
  dtResultsEntropy.foreach { case (param, auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%") }

  /* ... new cell ... */

  val dtResultsEntropy = Seq(1,2,3,4,5,10,20).map { param =>
    val model = trainDTWithParams(trainData, param, Gini)
    val scoreAndLabels = trainData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param tree depth", metrics.areaUnderROC)
  }
  dtResultsEntropy.foreach { case (param, auc) => println(f"$param,AUC = ${auc * 100}%2.2f%%") }

  /* ... new cell ... */

  val trainTestSplit = trainData.randomSplit(Array(0.6, 0.4), 123)
  val train = trainTestSplit(0)
  val test = trainTestSplit(1)

  /* ... new cell ... */

  val dtResultsTrain = Seq(1,2,3,4,5,10,20).map { param =>
  	val model = trainDTWithParams(train, param, Gini)
    val dtMetrics = Seq(model).map{ model =>
      val scoreAndLabels = test.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
    }
    dtMetrics
  }
  
  dtResultsTrain.map { mo => (mo(0)._1, mo(0)._2, mo(0)._3) }
}
              