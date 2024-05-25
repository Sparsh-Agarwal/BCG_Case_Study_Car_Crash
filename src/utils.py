import yaml


def load_csv(spark, file_path):
    """
    Read CSV file
    :param
        spark: spark instance
        file_path: csv file path
    :return: dataframe
    """
    return spark.read.option("inferSchema", "true") \
        .csv(file_path, header=True)


def read_yaml(file_path):
    """
    Read Config file in YAML format
    :param
        file_path: file path to config.yaml
    :return: dictionary with config details
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def write_csv(df, file_path):
    """
    Write CSV file
    :param
        df: dataframe
        file_path: output file path
    :return: None
    """
    df.repartition(1).write.format("csv") \
        .mode("overwrite").option("header", "true") \
        .save(file_path)
