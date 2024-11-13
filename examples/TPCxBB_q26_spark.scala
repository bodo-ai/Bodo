import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import scala.language.existentials

import org.apache.spark.sql.catalyst.analysis.UnresolvedRelation
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.execution.joins._

import org.apache.spark.ml.clustering.{KMeansModel, KMeans}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import scala.language.reflectiveCalls
import java.lang.management.ManagementFactory
import scala.collection.JavaConversions._

object Query26 {
  def main(args: Array[String]) {
    // Starting time
    val t0 = System.currentTimeMillis
    val spark = SparkSession
      .builder()
      .appName("Q26")
      .config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .getOrCreate()

    import spark.implicits._
    val table_store_sales_path = args(0)
    val table_item_path = args(1)

    val schema_store_sales = StructType(Array(
      StructField("ss_sold_date_sk", LongType, true),
      StructField("ss_sold_time_sk", LongType, true),
      StructField("ss_item_sk", LongType, true),
      StructField("ss_customer_sk", LongType, true),
      StructField("ss_cdemo_sk", LongType, true),
      StructField("ss_hdemo_sk", LongType, true),
      StructField("ss_addr_sk", LongType, true),
      StructField("ss_store_sk", LongType, true),
      StructField("ss_promo_sk", LongType, true),
      StructField("ss_ticket_number", LongType, true),
      StructField("ss_quantity", IntegerType, true),
      StructField("ss_wholesale_cost", FloatType, true),
      StructField("ss_list_price", FloatType, true),
      StructField("ss_sales_price", FloatType, true),
      StructField("ss_ext_discount_amt", FloatType, true),
      StructField("ss_ext_sales_price", FloatType, true),
      StructField("ss_ext_wholesale_cost", FloatType, true),
      StructField("ss_ext_list_price", FloatType, true),
      StructField("ss_ext_tax", FloatType, true),
      StructField("ss_coupon_amt", FloatType, true),
      StructField("ss_net_paid", FloatType, true),
      StructField("ss_net_paid_inc_tax", FloatType, true),
      StructField("ss_net_profit", FloatType, true)
      ))
    val df_store_sales = spark.read.schema(schema_store_sales).format("csv").option("sep", "|").load(table_store_sales_path)

    val schema_item = StructType(Array(
      StructField("i_item_sk", LongType, true),
      StructField("i_item_id", StringType, true),
      StructField("i_rec_start_date", StringType, true),
      StructField("i_rec_end_date", StringType, true),
      StructField("i_item_desc", StringType, true),
      StructField("i_current_price", FloatType, true),
      StructField("i_wholesale_cost", FloatType, true),
      StructField("i_brand_id", IntegerType, true),
      StructField("i_brand", StringType, true),
      StructField("i_class_id", IntegerType, true),
      StructField("i_class", StringType, true),
      StructField("i_category_id", IntegerType, true),
      StructField("i_category", StringType, true),
      StructField("i_manufact_id", IntegerType, true),
      StructField("i_manufact", StringType, true),
      StructField("i_size", StringType, true),
      StructField("i_formulation", StringType, true),
      StructField("i_color", StringType, true),
      StructField("i_units", StringType, true),
      StructField("i_container", StringType, true),
      StructField("i_manager_id", IntegerType, true),
      StructField("i_product_name", StringType, true)
      ))

    val df_item = spark.read.schema(schema_item).format("csv").option("sep", "|").load(table_item_path)
    // val df_item = spark.read.parquet(table_item_path)

    df_store_sales.registerTempTable("store_sales_table")
    df_item.registerTempTable("item_table")
    // collect() fails so using first()
    // df_store_sales.cache().first()
    // df_item.cache().first()

    val fin  = spark.sql("""
    SELECT
  ss.ss_customer_sk AS cid,
  count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,
  count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,
  count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,
  count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,
  count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,
  count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,
  count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7,
  count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,
  count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,
  count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,
  count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,
  count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS id12,
  count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,
  count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,
  count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15
FROM store_sales_table ss
INNER JOIN item_table i
  ON (ss.ss_item_sk = i.i_item_sk AND i.i_category = "Books"
  AND ss.ss_customer_sk IS NOT NULL
)
GROUP BY ss.ss_customer_sk
HAVING count(ss.ss_item_sk) > 5
    """)
    val assembler = new VectorAssembler().setInputCols(
      Array("id1", "id2", "id3","id4","id5","id6","id7",
      "id8","id9","id10","id11","id12","id13","id14","id15")).setOutputCol("features")
    val ds = assembler.transform(fin)
    ds.cache.first
    val t1 = System.currentTimeMillis

    // Measure time
    println("Query 26 time(s) took: " + (t1 - t0).toFloat / 1000)
  }
}
