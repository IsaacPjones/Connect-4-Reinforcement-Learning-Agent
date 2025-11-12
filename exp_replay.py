# inspiration from johnny code youtube https://www.youtube.com/watch?v=y3BSPfmMIkAa
from collections import deque
import random

class ReplayMemory():
  def __init__(self, max_len, seed=None):
    self.memory = deque([], maxlen = max_len)

    if seed is not None:
      random.seed(seed)
    
  def append(self, transition):
    self.memory.append(transition)
  
  def sample(self, sample_size):
    return random.sample(self.memory, sample_size)
  
  def __len__(self):
    return len(self.memory)