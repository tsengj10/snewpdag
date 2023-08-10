"""
Uniform - generate a constant rate of events to add to a TimeSeries or Hist1D
Arguments:
  field:  field specifier. Must be TimeSeries or Hist1D. Modified in place.
  rate:  number of events per second
  tmin:  (optional) starting time (seconds) of the generator.
  tmax:  (optional) stopping time (seconds).
  tnbins:  (optional) number of bins, if fixed Hist1D output.

If tmin/tmax are floats, use the number as a timestamp.
  If tmax < tmin, assume tmax is actually a time difference in seconds.
If tnbins > 0, assume we'll encounter a fixed Hist1D.
  Will just fill the first tnbins of the Hist1D. No limits checking
  (tmin/tmax optional)
If they're strings, interpret as a unix time string.
"""
import logging
import numpy as np
from astropy.time import Time

from snewpdag.dag import Node
from snewpdag.values import TimeSeries, Hist1D
from snewpdag.dag.lib import fetch_field

class Uniform (Node):
  def __init__(self, field, rate, tmin=None, tmax=None, tnbins=0, **kwargs):
    self.field = field
    self.rate = rate
    self.tmin = Time(tmin).to_value('unix', 'long') if isinstance(tmin, str) \
                else tmin
    self.tmax = Time(tmax).to_value('unix', 'long') if isinstance(tmax, str) \
                else tmax
    if self.tmin != None and self.tmax != None:
      if self.tmax < self.tmin:
        self.tmax = self.tmin + self.tmax # interpret tmax as a width
    self.tnbins = tnbins
    # if tnbins > 0, we expect a fixed histogram.
    # but this means we also must have tmin and tmax.
    super().__init__(**kwargs)

  def alert(self, data):
    v, flag = fetch_field(data, self.field)
    if flag:
      v = data[self.field]

      # suggested optimizations:
      # * Hist1D can be filled bin by bin with Poisson variates
      #   mu = self.rate * v.duration / v.nbins
      #   v.bins += Node.rng.poisson(mu, v.nbins)
      #   (but need to adjust means for partial bins at ends)
      # * for Hist1D, and TimeSeries with limits, can restrict generation
      #   to within those limits, rather than over the whole (tmin,tmax) 

      if isinstance(v, TimeSeries):
        t0 = v.start if self.tmin == None else self.tmin
        t1 = v.stop if self.tmax == None else self.tmax
        mean = (t1 - t0) * self.rate
        nev = Node.rng.poisson(mean) # Poisson fluctuations around mean
        if t0 == None or t1 == None:
          logging.error('{}: range unspecified for TimeSeries'.format(self.name))
          return False
        u = (t1 - t0) * Node.rng.random(size=nev, dtype=np.float64) + t0
        v.add(u)
        return True
      elif isinstance(v, Hist1D):
        if self.tnbins > 0:
          # fixed histogram. But we don't check limits, and just
          # fill at most the specified number of bins
          # (but at rate for specified number of bins)
          t0 = v.start if self.tmin == None else self.tmin
          t1 = v.stop if self.tmax == None else self.tmax
          mu = (t1 - t0) * self.rate / self.tnbins
          nb = self.tnbins if self.tnbins <= v.nbins else v.nbins
          v.bins[:nb] += Node.rng.poisson(mu, size=nb)
        else:
          # fill histogram as it's presented to us
          t0 = v.xlow if self.tmin == None else self.tmin
          t1 = v.xhigh if self.tmax == None else self.tmax
          j0 = v.bin_index(t0)
          j1 = v.bin_index(t1)
          # limits
          if j0 < 0:
            j0 = 0
            t0 = v.xlow
          if j1 >= v.nbins:
            j1 = v.nbins
            t1 = v.xhigh
          nb = j1 - j0
          rates = np.ones(nb)
          x0 = v.bin_edge(j0)
          x1 = v.bin_edge(j1)
          dt = v.granularity()
          if t0 != x0:
            rates[0] = 1.0 - (t0 - x0) / dt
          if not np.isclose(t1, x1):
            rates[-1] = (t1 - x1) / dt
          mu = dt * self.rate * rates
          v.bins += Node.rng.poisson(mu, size=nb)
          return True
      else:
        logging.error('{}: field is neither TimeSeries nor Hist1D'.format(self.name))
        return False
    else:
      return False
