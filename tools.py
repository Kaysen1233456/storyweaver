import multiprocessing
from tqdm import tqdm

def parallelize_with_multiprocessing(data_list, func, max_workers=4):
    # 适用于cpu密集性任务
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = []
        with tqdm(total=len(data_list)) as pbar:
            for result in pool.imap(func, data_list):
                results.append(result)
                pbar.update(1)
    return results