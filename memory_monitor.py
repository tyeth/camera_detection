#!/usr/bin/env python3
"""
Memory usage monitor for the Qualcomm device.
This script monitors memory usage during frame processing.
"""

import os
import time
import psutil
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def monitor_memory(duration=60, interval=1, plot=True, log_file=None):
    """
    Monitor memory usage for a specified duration.
    
    Args:
        duration: Monitoring duration in seconds
        interval: Sampling interval in seconds
        plot: Whether to generate a plot
        log_file: File path to save logs
    """
    timestamps = []
    memory_usage = []
    
    start_time = time.time()
    end_time = start_time + duration
    
    print(f"Monitoring memory usage for {duration} seconds...")
    
    try:
        while time.time() < end_time:
            current_time = time.time() - start_time
            memory_mb = get_memory_usage()
            
            timestamps.append(current_time)
            memory_usage.append(memory_mb)
            
            print(f"Time: {current_time:.1f}s, Memory: {memory_mb:.2f} MB")
            
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"{current_time:.1f},{memory_mb:.2f}\n")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    
    # Generate plot if requested
    if plot and timestamps:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_usage, marker='o')
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True)
        
        # Save plot
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"memory_usage_{timestamp_str}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        
        # Show plot if in interactive environment
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor memory usage of the current process")
    parser.add_argument("-d", "--duration", type=int, default=60,
                        help="Monitoring duration in seconds (default: 60)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting")
    parser.add_argument("-l", "--log-file", type=str,
                        help="File to save log data")
    
    args = parser.parse_args()
    
    # Create log file with header if specified
    if args.log_file:
        with open(args.log_file, 'w') as f:
            f.write("time_seconds,memory_mb\n")
    
    monitor_memory(
        duration=args.duration,
        interval=args.interval,
        plot=not args.no_plot,
        log_file=args.log_file
    )
