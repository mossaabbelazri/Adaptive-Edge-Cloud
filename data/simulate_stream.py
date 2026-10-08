import socket
import time
import json
import random
import math

def generate_summary_features(base_noise=0.1, wear_factor=0.0):
    """
    Simulates the statistical features (RMS, Peak, Variance) 
    extracted from the raw vibration data, exactly matching 
    what the Neural Network was trained on.
    """
    # Base values for a healthy bearing
    rms_base = 0.07
    peak_base = 0.1
    var_base = 0.005
    
    # Add wear factor (noise and amplitude increase)
    rms = abs(random.gauss(rms_base + (wear_factor * 0.1), base_noise + wear_factor))
    peak = abs(random.gauss(peak_base + (wear_factor * 0.3), base_noise + wear_factor))
    var = abs(random.gauss(var_base + (wear_factor * 0.05), base_noise + wear_factor))
    
    # Match the exact 10 features extracted in the Colab training script
    return [
        rms, peak, var, 
        rms*1.1, peak*0.9, var*1.05, 
        rms*0.9, peak*1.1, var*0.95, 
        rms
    ]

def start_stream_server(host='localhost', port=9999):
    """
    Simulates the NASA Bearing Dataset telemetry stream.
    Features represent vibration data across 4 bearings.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"[NASA Bearing Simulator] Started on {host}:{port}")
    print("[NASA Bearing Simulator] Waiting for PySpark to connect...")
    
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    
    # Simulating progressive degradation (like a failing bearing)
    degradation_step = 0.005
    current_wear = 0.0
    
    try:
        batch_count = 0
        while True:
            # The Colab models were trained on the statistical features of a failing bearing.
            # We generate those exact 10 features here, increasing the wear over time.
            features = generate_summary_features(base_noise=0.01, wear_factor=current_wear)
            
            data = {
                "timestamp": time.time(),
                "sensor_id": "nasa_bearing_test_rig",
                "features": features
            }
            
            msg = json.dumps(data) + '\n'
            conn.sendall(msg.encode('utf-8'))
            
            # Increase wear over time to trigger uncertainty in the Edge Model
            current_wear += degradation_step
            batch_count += 1
            
            if batch_count % 50 == 0:
                print(f"[Simulator] Bearing 2 wear level: {current_wear:.2f}")
                
            time.sleep(0.05) # Fast stream
            
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. PySpark stream likely stopped.")
    finally:
        conn.close()
        server_socket.close()
        print("Server shutdown.")

if __name__ == "__main__":
    start_stream_server()
