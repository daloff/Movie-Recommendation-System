
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.MinHashLSH;
import org.apache.spark.ml.feature.MinHashLSHModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class SparkRecommendationSystemNNS {

	// change this to your own file path
	private static final String FILE_URI = "file:///C:/dev/datasets/doc_*.txt";

	public static void main(String[] args) {

		// initializing spark
		SparkSession spark = SparkSession.builder().config("spark.master", "local[*]").getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
		sc.setLogLevel("WARN");

		// create RDD by reading text files
		JavaPairRDD<String, String> documents = sc.wholeTextFiles(FILE_URI);
		System.out.println(documents.take((int) documents.count()).toString());

		// break each document into words
		JavaPairRDD<Tuple2<String, String[]>, Long> wDocuments = documents.mapValues(new Function<String, String[]>() {
			public String[] call(String line) throws Exception {
				return line.split("\\W+"); // use the following for English
				// return line.split("\\|"); // use the following for Chinese
			}
		}).zipWithIndex();
		System.out.println(wDocuments.take((int) wDocuments.count()).toString());

		// load wDocuments into dataframe
		StructType schema = new StructType(
				new StructField[] { DataTypes.createStructField("docID", DataTypes.LongType, false),
						DataTypes.createStructField("file_path", DataTypes.StringType, false),
						DataTypes.createStructField("all_words", DataTypes.createArrayType(DataTypes.StringType, false),
								false) });
		Dataset<Row> documentsWithAllWords = spark
				.createDataFrame(wDocuments.map(new Function<Tuple2<Tuple2<String, String[]>, Long>, Row>() {
					@Override
					public Row call(Tuple2<Tuple2<String, String[]>, Long> record) {
						return RowFactory.create(record._2(),
								record._1()._1().substring(record._1._1().lastIndexOf("/") + 1), record._1()._2());
					}
				}), schema);
		documentsWithAllWords.show(true);

		// remove stop words
		StopWordsRemover remover = new StopWordsRemover().setInputCol("all_words").setOutputCol("words");
		Dataset<Row> documentsWithoutStopWords = remover.transform(documentsWithAllWords).select("docID", "file_path",
				"words");
		documentsWithoutStopWords.show(true);

		// fit a CountVectorizerModel from the corpus
		CountVectorizer vectorizer = new CountVectorizer().setInputCol("words").setOutputCol("TF_values");
		CountVectorizerModel cvm = vectorizer.fit(documentsWithoutStopWords);
		System.out.println("vocab size = " + cvm.vocabulary().length);
		for (int i = 0; i < cvm.vocabulary().length; i++) {
			System.out.print(cvm.vocabulary()[i] + "(" + i + ") ");
		}
		System.out.println();
		Dataset<Row> tf = cvm.transform(documentsWithoutStopWords);
		tf.show(true);

		// Normalize each Vector using L1 norm.
		Normalizer normalizer = new Normalizer().setInputCol("TF_values").setOutputCol("normalized_TF").setP(1.0);
		Dataset<Row> normalizedTF = normalizer.transform(tf);
		normalizedTF.show(true);

		// calcualte TF-IDF values
		IDF idf = new IDF().setInputCol("normalized_TF").setOutputCol("TFIDF_values");
		IDFModel idfModel = idf.fit(normalizedTF);
		Dataset<Row> tf_idf = idfModel.transform(normalizedTF);
		tf_idf.select("docID", "file_path", "words", "TFIDF_values").show(true);

		// find similar documents
		JavaRDD<IndexedRow> rddIndexRows = tf_idf.toJavaRDD().map(new Function<Row, IndexedRow>() {
			public IndexedRow call(Row row) throws Exception {
				Object features = row.getAs("TFIDF_values");
				org.apache.spark.ml.linalg.DenseVector dense = null;

				if (features instanceof org.apache.spark.ml.linalg.DenseVector) {
					dense = (org.apache.spark.ml.linalg.DenseVector) features;
				} else if (features instanceof org.apache.spark.ml.linalg.SparseVector) {
					org.apache.spark.ml.linalg.SparseVector sparse = (org.apache.spark.ml.linalg.SparseVector) features;
					dense = sparse.toDense();
				} else {
					RuntimeException e = new RuntimeException(
							"Cannot convert to " + features.getClass().getCanonicalName());
					throw e;
				}
				org.apache.spark.mllib.linalg.Vector vec = org.apache.spark.mllib.linalg.Vectors.dense(dense.toArray());
				return new IndexedRow((long) row.getAs("docID"), vec);
			}
		});
		System.out.println(rddIndexRows.take((int) rddIndexRows.count()).toString());

		IndexedRowMatrix irm = new IndexedRowMatrix(rddIndexRows.rdd());
		irm = irm.toBlockMatrix().transpose().toIndexedRowMatrix();
		System.out.println("feature matrix has " + irm.numRows() + " rows, and " + irm.numCols() + " columns");
		CoordinateMatrix cosineSimilarities = irm.columnSimilarities();
		System.out.println("final matrix has " + cosineSimilarities.numRows() + " rows, and "
				+ cosineSimilarities.numCols() + " columns");

		JavaRDD<MatrixEntry> results = cosineSimilarities.entries().toJavaRDD();
		System.out.println(results.take((int) results.count()).toString());

		// showing the final results. How to read this result is discussed in the class
		JavaRDD<String> resultRDD = results.map(new Function<MatrixEntry, String>() {
			public String call(MatrixEntry e) {
				return String.format("%d,%d,%s", e.i(), e.j(), e.value());
			}
		});

		spark.close();
	}

}
