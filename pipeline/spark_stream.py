import json
import torch
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType

import sys
import os

# Add parent directory to path to import models and controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edge_model import get_quantized_edge_model
from models.cloud_model import get_cloud_model
from controller.uncertainty import calculate_entropy, is_uncertain
from controller.mape_k import MAPEKLoop

# Initialize models and controller on the driver
edge_model = get_quantized_edge_model()
cloud_model = get_cloud_model()
controller = MAPEKLoop(initial_threshold=0.8)

# Assuming models are trained and checkpoints exist for demo purposes.
# We will use random initialization if checkpoints are not found (acceptable for structure demo)
if os.path.exists('checkpoints/edge_model.pth'):
    edge_model.load_state_dict(torch.load('checkpoints/edge_model.pth'))
if os.path.exists('checkpoints/cloud_model.pth'):
    cloud_model.load_state_dict(torch.load('checkpoints/cloud_model.pth'))

def process_batch(df, epoch_id):
    """
    Function applied to each micro-batch in the stream.
    Executes inference and self-adaptation logic.
    """
    # Convert PySpark DataFrame to Pandas for easier PyTorch integration in this demo
    pdf = df.toPandas()
    if pdf.empty:
        return

    # Extract features
    features_list = pdf['features'].tolist()
    if not features_list:
        return
        
    import numpy as np
    X_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)

    # 1. Edge Inference (All data)
    with torch.no_grad():
        edge_logits = edge_model(X_tensor)
        edge_entropy = calculate_entropy(edge_logits, features=X_tensor)

    # 2. Uncertainty Evaluation based on dynamic threshold
    uncertain_mask = is_uncertain(edge_entropy, threshold=controller.threshold)
    uncertain_indices = torch.nonzero(uncertain_mask).squeeze(-1)
    
    total_samples = len(X_tensor)
    uncertain_samples_count = len(uncertain_indices) if uncertain_indices.dim() > 0 else (1 if uncertain_mask.item() else 0)

    if uncertain_indices.dim() == 0 and uncertain_mask.item():
        uncertain_indices = torch.tensor([0]) # Handle single element batch case
    elif uncertain_indices.dim() == 0:
         uncertain_indices = torch.tensor([], dtype=torch.long)

    # 3. Cloud Offloading (Only for uncertain data)
    if len(uncertain_indices) > 0:
        uncertain_data = X_tensor[uncertain_indices]
        with torch.no_grad():
            cloud_logits = cloud_model(uncertain_data)
        print(f"Batch {epoch_id}: Offloaded {len(uncertain_indices)}/{total_samples} samples to Cloud.")
    else:
        print(f"Batch {epoch_id}: Processed entirely on Edge ({total_samples} samples).")

    # 4. Self-Adaptation (MAPE-K Loop)
    controller.step(total_samples, uncertain_samples_count)

def start_pipeline():
    """
    Initializes and starts the PySpark Structured Streaming pipeline.
    """
    spark = SparkSession.builder \
        .appName("AdaptiveEdgeCloudPipeline") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    # Define schema matching the simulator output
    schema = StructType([
        StructField("timestamp", DoubleType(), True),
        StructField("sensor_id", StringType(), True),
        StructField("features", ArrayType(DoubleType()), True)
    ])

    print("Connecting to data stream at localhost:9999...")
    
    # Read from socket
    raw_stream = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    # Parse JSON
    parsed_stream = raw_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Process each micro-batch
    query = parsed_stream.writeStream \
        .foreachBatch(process_batch) \
        .start()

    print("Pipeline running. Awaiting data...")
    query.awaitTermination()

if __name__ == "__main__":
    start_pipeline()
