import socket
import time
import json
import random
import math

def generate_bearing_vibration(base_noise, wear_factor):
    """
    Simulates a vibration signal for a bearing. 
    As the wear_factor increases, the amplitude and noise increase.
    """
    # Simulate a 2000 RPM (approx 33.3 Hz) base frequency + harmonics
    t = time.time()
    signal = math.sin(2 * math.pi * 33.3 * t) + 0.5 * math.sin(2 * math.pi * 66.6 * t)
    
    # Add noise based on wear and tear
    noise = random.gauss(0, base_noise + wear_factor)
    return signal * (1.0 + wear_factor) + noise

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
            # We have 4 bearings, we generate synthetic features for each (e.g., 2 axes each = 8 features) + 2 metadata features = 10 total
            # Bearing 1 is healthy
            # Bearing 2 is degrading over time
            
            b1_vib = generate_bearing_vibration(base_noise=0.1, wear_factor=0.0)
            b2_vib = generate_bearing_vibration(base_noise=0.1, wear_factor=current_wear)
            
            # Fill the 10 features expected by our PyTorch model
            features = [
                b1_vib, b1_vib * 0.9,  # Bearing 1 (X, Y)
                b2_vib, b2_vib * 1.1,  # Bearing 2 (X, Y) - Degrading
                generate_bearing_vibration(0.1, 0.0), generate_bearing_vibration(0.1, 0.0), # Bearing 3
                generate_bearing_vibration(0.1, 0.0), generate_bearing_vibration(0.1, 0.0), # Bearing 4
                2000.0 + random.uniform(-10, 10), # RPM 
                6000.0 + random.uniform(-50, 50)  # Load
            ]
            
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
