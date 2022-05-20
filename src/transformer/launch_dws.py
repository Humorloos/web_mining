import logging
import os
from subprocess import Popen, PIPE
from time import sleep
from io import StringIO

import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %%
while True:
    gpu_data = pd.read_csv(StringIO(
        Popen(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv'], stdout=PIPE)
            .communicate()[0]
            .decode('utf-8')
    )).rename(columns={'utilization.gpu [%]': 'gpu', ' memory.free [MiB]': 'memory'})
    print(gpu_data)
    gpu_data.gpu = gpu_data.gpu.str.rstrip(' %').astype('int32')
    gpu_data.memory = gpu_data.memory.str.rstrip(' MiB').astype('int32')
    # For training on GPU with largest free memory
    target_gpu = gpu_data.sort_values(by='memory', ascending=False).index[0]
    logging.info(f'Starting training on GPU f{target_gpu}')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
    try:
        from tune import *
    except Exception as e:
        while True:
            logging.exception(e)
            sleep(60*60*24*7)
    break
    # For waiting for some GPU to get idle
    # idle_gpus = gpu_data[(gpu_data.gpu == 0) & (gpu_data.memory > 48e3)]
    # if idle_gpus.shape[0] > 0:
    #     target_gpu = idle_gpus.index[0]
    #     logging.info(f'Found GPU {target_gpu} to be idle, starting training on GPU {target_gpu}')
    #     os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
    #     try:
    #         from tune import *
    #     except Exception as e:
    #         while True:
    #             logging.exception(e)
    #             sleep(60*60*24*7)
    #     break
    # else:
    #     logging.info('All GPUs are occupied, sleeping for 5 minutes')
    #     sleep(5 * 60)
