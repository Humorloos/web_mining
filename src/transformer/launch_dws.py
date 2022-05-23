import logging
import os
from time import sleep

from utils import get_idle_gpus

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# %%
while True:
    # # For training on GPU with largest free memory
    # target_gpu = get_gpu_with_most_available_memory()
    # logging.info(f'Starting training on GPU {target_gpu}')
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
    # For training on idle GPUs
    target_gpu = ','.join(str(gpu) for gpu in get_idle_gpus())
    logging.info(f'Starting training on GPU(s) {target_gpu}')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)
    try:
        from tune import *
    except Exception as e:
        while True:
            logging.exception(e)
            sleep(60 * 60 * 24 * 7)
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
