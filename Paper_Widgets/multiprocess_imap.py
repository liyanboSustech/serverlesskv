import multiprocessing
from functools import partial

def process_item(item, multiplier):
    # Simulate a processing task
    return item * multiplier

if __name__ == "__main__":
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=4) as pool:
        # Create an iterable of items to process
        items = [1, 2, 3, 4, 5]
        
        # Partial function with a fixed multiplier argument
        func = partial(process_item, multiplier=10)
        
        # Use imap to process items in parallel
        results = pool.imap(func, items)
        
        # Iterate over results as they become available
        for result in results:
            print(result)
