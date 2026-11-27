"""Small owner-thread stage profiler; inclusive nested wall/CPU timings."""
from functools import wraps
from time import perf_counter,process_time
from contextlib import contextmanager

@contextmanager
def measure(counters,stage):
 wall,cpu=perf_counter(),process_time()
 try:yield
 finally:
  elapsed=perf_counter()-wall
  counters[stage+'_wall_s']+=elapsed;counters[stage+'_cpu_s']+=process_time()-cpu
  counters[stage+'_calls']+=1;counters[stage+'_max_s']=max(counters[stage+'_max_s'],elapsed)

def measured(stage):
 def decorate(fn):
  @wraps(fn)
  def wrap(self,*args,**kwargs):
   with measure(self.processing,stage):return fn(self,*args,**kwargs)
  return wrap
 return decorate

def phase_cycles(frequency_hz, sample, sample_rate):
    """Exact modular start phase for a float frequency and integer sample index.

    Convert only the reduced fraction to float; no huge phase or timestamp float.
    """
    numerator,denominator=float(frequency_hz).as_integer_ratio()
    modulus=denominator*int(sample_rate)
    return ((numerator*int(sample)) % modulus)/modulus
