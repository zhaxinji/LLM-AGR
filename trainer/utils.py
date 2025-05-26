
import torch
import random
import numpy as np
from logging import getLogger

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DisabledSummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper