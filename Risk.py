from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, max as spark_max, log,lag
from pyspark.sql.window import Window
import os

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Data Cleaning and Risk Identification") \
    .getOrCreate()

# 数据集路径
base_path = "/root/autodl-tmp/Dataset/"

# 获取所有文件夹
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f != 'risk_analysis_results']

# 数据清洗和风险识别函数
def process_data(folder):
    # 读取 CSV 文件
    file_path = os.path.join(base_path, folder, f"{folder}_data.csv")
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # 数据清洗
    # 1. 处理缺失值：填充或删除
    df = df.na.fill({"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0})

    # 2. 过滤无效数据：例如收盘价为负数的数据
    df = df.filter(col("close") > 0)

    # 3. 标准化日期格式（假设日期格式为 yyyy-MM-dd）
    df = df.withColumn("date", col("date").cast("date"))

    window_spec = Window.partitionBy("name").orderBy("close")
    # 计算对数收益率
    df = df.withColumn("prev_close", lag("close").over(window_spec))
    df = df.withColumn("log_return", log(col("close") / col("prev_close")))


    # 风险识别
    # 1. 计算波动率（基于收盘价的标准差）
    volatility = df.groupBy("name").agg(stddev("log_return").alias("volatility")).first()['volatility']

    # 2. 计算历史最大回撤和平均回撤
    window_spec = Window.orderBy("date")
    df = df.withColumn("max_close", spark_max("close").over(window_spec))
    df = df.withColumn("drawdown", (col("max_close") - col("close")) / col("max_close"))

    # 历史最大回撤
    max_drawdown = df.select(spark_max("drawdown").alias("max_drawdown")).collect()[0]["max_drawdown"]

    # 平均回撤
    avg_drawdown = df.select(mean("drawdown").alias("avg_drawdown")).collect()[0]["avg_drawdown"]

    # 3. 计算滚动回撤（过去 30 天的最大回撤）
    rolling_window_spec = Window.orderBy("date").rowsBetween(-30, 0)
    df = df.withColumn("rolling_max_close", spark_max("close").over(rolling_window_spec))
    df = df.withColumn("rolling_drawdown", (col("rolling_max_close") - col("close")) / col("rolling_max_close"))
    max_rolling_drawdown = df.select(spark_max("rolling_drawdown").alias("max_rolling_drawdown")).collect()[0]["max_rolling_drawdown"]

    # 4. 计算条件回撤（市场下跌期间的回撤）
    market_down_df = df.filter(col("close") < col("open"))  # 筛选市场下跌的数据
    conditional_drawdown = market_down_df.select(mean("drawdown").alias("conditional_drawdown")).collect()[0]["conditional_drawdown"]

    # 5. 计算平均交易量
    avg_volume = df.select(mean("volume").alias("avg_volume")).collect()[0]["avg_volume"]

    # 标记高风险数据
    risk_threshold_volatility = 0.3  # 波动率阈值
    risk_threshold_drawdown = 0.3  # 最大回撤阈值
    risk_threshold_rolling_drawdown = 0.25  # 滚动回撤阈值

    is_high_risk = (
        (volatility > risk_threshold_volatility) or
        (max_drawdown > risk_threshold_drawdown) or
        (max_rolling_drawdown > risk_threshold_rolling_drawdown)
    )

    # 返回结果
    return {
        "name": folder,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "max_rolling_drawdown": max_rolling_drawdown,
        "conditional_drawdown": conditional_drawdown,
        "avg_volume": avg_volume,
        "is_high_risk": is_high_risk
    }

# 处理所有文件夹中的数据
results = []
for folder in folders:
    result = process_data(folder)
    results.append(result)

# 将结果转换为 DataFrame 并保存
results_df = spark.createDataFrame(results)
output_path = os.path.join(base_path, "risk_analysis_results")
results_df.write.mode("overwrite").csv(output_path, header=True)

# 打印结果
results_df.show()

# 停止 SparkSession
spark.stop()