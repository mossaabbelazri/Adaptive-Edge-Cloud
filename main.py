import multiprocessing
import time
import os
import sys

# Set HADOOP_HOME for PySpark on Windows
hadoop_home = r'C:\Users\Lenovo\Desktop\MASTER\Projet'
os.environ['HADOOP_HOME'] = hadoop_home
# Add the bin directory to PATH so Java can find hadoop.dll
os.environ['PATH'] = hadoop_home + r'\bin;' + os.environ.get('PATH', '')

from data.simulate_stream import start_stream_server
from pipeline.spark_stream import start_pipeline
from models.train import train_and_save

def run_server():
    start_stream_server()

def run_pipeline():
    start_pipeline()

if __name__ == '__main__':
    print("==================================================")
    print("   Adaptive Edge-Cloud Inference Pipeline Demo    ")
    print("==================================================")
    
    # Ensure checkpoints exist
    if not os.path.exists('checkpoints/edge_model.pth') or not os.path.exists('checkpoints/cloud_model.pth'):
        train_and_save()

    print("\nStarting Data Stream Simulator in background...")
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    # Give server a moment to start
    time.sleep(2)

    print("\nStarting PySpark Pipeline...")
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=1)
        print("System shutdown complete.")

