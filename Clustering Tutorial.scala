
object Cells {
  val movies = sc.textFile("/home/david/spark/ml-100k/u.item")
  println(movies.first)

  /* ... new cell ... */

  val genres = sc.textFile("/home/david/spark/ml-100k/u.genre")
  genres.take(5).foreach(println)

  /* ... new cell ... */

  val genreMap = genres.filter(!_.isEmpty).map(line => line.
  split("\\|")).map(array => (array(1), array(0))).collectAsMap
  println(genreMap)

  /* ... new cell ... */

  val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
  val genres = array.toSeq.slice(5, array.size)
  val genresAssigned = genres.zipWithIndex.filter { case (g, idx) =>
  g == "1"
  }.map { case (g, idx) =>
  genreMap(idx.toString)
  }
  (array(0).toInt, (array(1), genresAssigned))
  }
  println(titlesAndGenres.first)

  /* ... new cell ... */

  import org.apache.spark.mllib.recommendation.ALS
  import org.apache.spark.mllib.recommendation.Rating
  val rawData = sc.textFile("/home/david/spark/ml-100k/u.data")
  val rawRatings = rawData.map(_.split("\t").take(3))
  val ratings = rawRatings.map{ case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
  ratings.cache
  val alsModel = ALS.train(ratings, 50, 10, 0.1)

  /* ... new cell ... */

  import org.apache.spark.mllib.linalg.Vectors
  val movieFactors = alsModel.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
  val movieVectors = movieFactors.map(_._2)
  val userFactors = alsModel.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
  val userVectors = userFactors.map(_._2)

  /* ... new cell ... */

  import org.apache.spark.mllib.clustering.KMeans
  val numClusters = 5
  val numIterations = 10
  val numRuns = 3
  val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
  
  // train user model
  val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)

  /* ... new cell ... */

  // predict a movie cluster for movie 1
  val movie1 = movieVectors.first
  val movieCluster = movieClusterModel.predict(movie1)
  println(movieCluster)
  // 4
  // predict clusters for all movies
  val predictions = movieClusterModel.predict(movieVectors)
  println(predictions.take(10).mkString(","))
  // 0,0,1,1,2,1,0,1,1,1

  /* ... new cell ... */

  // inspect the movie clusters, by looking at the movies that are closest to each cluster center
  
  // define Euclidean distance function
  import breeze.linalg._
  import breeze.numerics.pow
  def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum
  
  // join titles with the factor vectors, and compute the distance of each vector from the assigned cluster center
  val titlesWithFactors = titlesAndGenres.join(movieFactors)
  val moviesAssigned = titlesWithFactors.map { case (id, ((title, genres), vector)) => 
  	val pred = movieClusterModel.predict(vector)
  	val clusterCentre = movieClusterModel.clusterCenters(pred)
  	val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
  	(id, title, genres.mkString(" "), pred, dist) 
  }
  val clusterAssignments = moviesAssigned.groupBy { case (id, title, genres, cluster, dist) => cluster }.collectAsMap 
  
  for ( (k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
  	println(s"Cluster $k:")
  	val m = v.toSeq.sortBy(_._5)
  	println(m.take(20).map { case (_, title, genres, _, d) => (title, genres, d) }.mkString("\n")) 
  	println("=====\n")
  }

  /* ... new cell ... */

  // clustering mathematical evaluation
  
  // compute the cost (WCSS) on for movie and user clustering
  val movieCost = movieClusterModel.computeCost(movieVectors)
  val userCost = userClusterModel.computeCost(userVectors)
  println("WCSS for movies: " + movieCost)
  println("WCSS for users: " + userCost)

  /* ... new cell ... */

  // cross-validation for movie clusters
  val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
  val trainMovies = trainTestSplitMovies(0)
  val testMovies = trainTestSplitMovies(1)
  val costsMovies = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainMovies, numIterations, k, numRuns).computeCost(testMovies)) }
  println("Movie clustering cross-validation:")
  costsMovies.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }

  /* ... new cell ... */

  // cross-validation for user clusters
  val trainTestSplitUsers = userVectors.randomSplit(Array(0.6, 0.4), 123)
  val trainUsers = trainTestSplitUsers(0)
  val testUsers = trainTestSplitUsers(1)
  val costsUsers = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainUsers, numIterations, k, numRuns).computeCost(testUsers)) }
  println("User clustering cross-validation:")
  costsUsers.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
}
              