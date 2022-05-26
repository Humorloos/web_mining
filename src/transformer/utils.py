import logging
from datetime import datetime, timezone, timedelta
from io import StringIO
from subprocess import Popen, PIPE

import pandas as pd
import torch


def get_timestamp():
    return pd.Timestamp.today(
        tz=datetime.now(timezone(timedelta(0))).astimezone().tzinfo
    ).strftime('%Y-%m-%d_%H.%M')


def get_gpu_with_most_available_memory():
    """
    Gets the gpu with the largest free memory.
    :return: the gpu's index
    """
    gpu_data = get_nvidia_smi_data('--query-gpu=index,utilization.gpu,memory.free')\
        .rename(columns={' utilization.gpu [%]': 'gpu', ' memory.free [MiB]': 'memory'})
    print(gpu_data)
    gpu_data.gpu = gpu_data.gpu.str.rstrip(' %').astype('int32')
    gpu_data.memory = gpu_data.memory.str.rstrip(' MiB').astype('int32')
    available_devices = set()
    # Prioritize previously used GPUs
    for i in gpu_data['index']:
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
            available_devices.add(i)
        except Exception as e:
            logging.exception(e)
    return str(gpu_data[gpu_data['index'].isin(available_devices)]
               .sort_values(by='memory', ascending=False)['index'][0])


def get_idle_gpus():
    """
    Gets a list of the indexes of all gpus that are currently idle (not running any process)
    :return: list of the gpus' indexes
    """
    gpu_data = get_nvidia_smi_data('--query-gpu=index,gpu_uuid').rename(columns={' uuid': 'uuid'})
    gpu_data.uuid = gpu_data.uuid.str.lstrip(' ')
    process_data = get_nvidia_smi_data('--query-compute-apps=gpu_uuid,process_name')
    occupied_gpus = set(process_data['gpu_uuid'])
    return gpu_data.loc[~gpu_data['uuid'].isin(occupied_gpus), 'index'].tolist()


def get_nvidia_smi_data(query_flag):
    """
    Gets a dataframe containing information retrieved via the 'nvidia-smi' command
    :param query_flag: query flag for the nvidia-smi command
    (see https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf)
    :return: a dataframe with the information returned by the nvidia-smi call
    """
    return pd.read_csv(StringIO(
        Popen(['nvidia-smi', query_flag, '--format=csv'], stdout=PIPE)
            .communicate()[0]
            .decode('utf-8')
    ))
