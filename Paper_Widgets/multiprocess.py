import multiprocessing
import time

def worker(n):
    print(f"Worker {n} started")
    time.sleep(2)  # Simulate a time-consuming task
    print(f"Worker {n} finished")

if __name__ == "__main__":
    with multiprocessing.Pool(processes=1) as pool:
        pool.map(worker, range(8))  # Launch 8 tasks with 4 processes
