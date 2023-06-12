import os
from enum import Enum
from .device_id import DeviceId

#

#NOTE:  在使用torch之前必须调用这个函数

class DeviceException(Exception):
    pass

class _Device:
    def __init__(self):
        self.set(DeviceId.CPU)

    def is_gpu(self):
        ''' Returns `True` if the current device is GPU, `False` otherwise. '''
        return self.current() is not DeviceId.CPU
  
    def current(self):
        return self._current_device

    def set(self, device:DeviceId):     
        if device == DeviceId.CPU:
            os.environ['CUDA_VISIBLE_DEVICES']=''
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=str(device.value)
            import torch
            torch.backends.cudnn.benchmark=False
        
        self._current_device = device    
        return device