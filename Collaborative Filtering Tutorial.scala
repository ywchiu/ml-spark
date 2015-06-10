
object Cells {
  val rawData = sc.textFile("/home/david/spark/ml-100k/u.data")
  rawData.first() 

  /* ... new cell ... */

  val rawRatings = rawData.map(_.split("\t").take(3))

  /* ... new cell ... */

  import org.apache.spark.mllib.recommendation.ALS
  import org.apache.spark.mllib.recommendation.Rating
  
  val ratings = rawRatings.map { case Array(user, movie, rating) =>
  Rating(user.toInt, movie.toInt, rating.toDouble) }

  /* ... new cell ... */

  val model = ALS.train(ratings, 50, 10, 0.01)

  /* ... new cell ... */

  model.productFeatures

  /* ... new cell ... */

  model.userFeatures.count

  /* ... new cell ... */

  println(model.userFeatures.count)
  println(model.productFeatures.count)

  /* ... new cell ... */

  val predictedRating = model.predict(789, 123)

  /* ... new cell ... */

  val userId = 789
  val K = 10
  val topKRecs = model.recommendProducts(userId, K)

  /* ... new cell ... */

  val movies = sc.textFile("/home/david/spark/ml-100k/u.item")
  val titles = movies.map(line => line.split("\\|").take(2)).map(array
  => (array(0).toInt, array(1))).collectAsMap()
  titles(123)

  /* ... new cell ... */

  val moviesForUser = ratings.keyBy(_.user).lookup(789)
  println(moviesForUser.size)

  /* ... new cell ... */

  moviesForUser.sortBy(-_.rating).take(10).map(rating => 
  (titles(rating.product), rating.rating)).foreach(println)
  
  topKRecs.map(rating => (titles(rating.product), rating.rating)).
  foreach(println)

  /* ... new cell ... */

  import org.jblas.DoubleMatrix
  val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
  
  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double =
  {
  vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

  /* ... new cell ... */

  val itemId = 567
  val itemFactor = model.productFeatures.lookup(itemId).head
  val itemVector = new DoubleMatrix(itemFactor)
  cosineSimilarity(itemVector, itemVector)

  /* ... new cell ... */

  val sims = model.productFeatures.map{ 
    case (id, factor) =>
  val factorVector = new DoubleMatrix(factor)
  val sim = cosineSimilarity(factorVector, itemVector)
  (id, sim)
  }
  
  val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { 
    case(id, similarity) => similarity })

  /* ... new cell ... */

  println(sortedSims.take(10).mkString("\n"))

  /* ... new cell ... */

  val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] {
  case (id, similarity) => similarity })
  sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim)
  }.mkString("\n")

  /* ... new cell ... */

  val actualRating = moviesForUser.take(10){0}
  val predictedRating = model.predict(789, actualRating.product)
  val squaredError = math.pow(predictedRating - actualRating.rating,2.0)

  /* ... new cell ... */

  val usersProducts = ratings.map{ case Rating(user, product, rating) => (user, product)}
  val predictions = model.predict(usersProducts).map{
  case Rating(user, product, rating) => ((user, product), rating)
  }

  /* ... new cell ... */

  val ratingsAndPredictions = ratings.map{
  case Rating(user, product, rating) => ((user, product), rating)
  }.join(predictions)

  /* ... new cell ... */

  val MSE = ratingsAndPredictions.map{
  case ((user, product), (actual, predicted)) => math.pow((actual -
  predicted), 2)
  }.reduce(_ + _) / ratingsAndPredictions.count
  println("Mean Squared Error = " + MSE)

  /* ... new cell ... */

  import org.apache.spark.mllib.evaluation.RegressionMetrics
  val predictedAndTrue = ratingsAndPredictions.map { case ((user,
  product), (predicted, actual)) => (predicted, actual) }
  val regressionMetrics = new RegressionMetrics(predictedAndTrue)
  
  println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
  println("Root Mean Squared Error = " + regressionMetrics.
  rootMeanSquaredError)

  /* ... new cell ... */

  val predictedMovies = topKRecs.map(_.product)
  val predK = predictedMovies.take(10)
  var score = 0.0
  var numHits = 0.0

  /* ... new cell ... */

  val actual = moviesForUser.map(_.product)
  for ((p, i) <- predK.zipWithIndex) {
  if (actual.contains(p)) {
  numHits += 1.0
  score += numHits / (i.toDouble + 1.0)
    println
  }
  }
  predK.zipWithIndex

  /* ... new cell ... */

  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int):
  Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
    1.0
    } else {
    score / scala.math.min(actual.size, k).toDouble
    }
  }

  /* ... new cell ... */

  val actualMovies = moviesForUser.map(_.product)
  val predictedMovies = topKRecs.map(_.product)
  val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)

  /* ... new cell ... */

  val predictedMovies = topKRecs.map(_.product)

  /* ... new cell ... */

  val itemFactors = model.productFeatures.map { case (id, factor) =>
  factor }.collect()
  val itemMatrix = new DoubleMatrix(itemFactors)
  println(itemMatrix.rows, itemMatrix.columns)
  //將item matrix broadcast 給每個node
  val imBroadcast = sc.broadcast(itemMatrix)

  /* ... new cell ... */

  val allRecs = model.userFeatures.map{ case (userId, array) =>
  val userVector = new DoubleMatrix(array)
  val scores = imBroadcast.value.mmul(userVector)
  val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
  val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
  (userId, recommendedIds)
  }

  /* ... new cell ... */

  val userMovies = ratings.map{ case Rating(user, product, rating) =>
  (user, product) }.groupBy(_._1)
  
  val K = 10
  val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted,
  actualWithIds)) =>
  val actual = actualWithIds.map(_._2).toSeq
  avgPrecisionK(actual, predicted, K)
  }.reduce(_ + _) / allRecs.count
  println("Mean Average Precision at K = " + MAPK)

  /* ... new cell ... */

  val userMovies = ratings.map{ case Rating(user, product, rating) =>
  (user, product) }.groupBy(_._1)
  
  val K = 2000
  val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted,
  actualWithIds)) =>
  val actual = actualWithIds.map(_._2).toSeq
  avgPrecisionK(actual, predicted, K)
  }.reduce(_ + _) / allRecs.count
  println("Mean Average Precision at K = " + MAPK)

  /* ... new cell ... */

  import org.apache.spark.mllib.evaluation.RankingMetrics
  val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case
  (userId, (predicted, actualWithIds)) =>
  val actual = actualWithIds.map(_._2)
  (predicted.toArray, actual.toArray)
  }
  val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
  println("Mean Average Precision = " + rankingMetrics.
  meanAveragePrecision)
}
              