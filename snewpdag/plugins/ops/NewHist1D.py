"""
NewHist1D - initialize an empty Hist1D object

Arguments:
  out_field:  output field name
  nbins:  number of bins
  start:  float to indicate start time (if this was a time histogram)
  stop:  float to indicate stop time

If start or stop are strings, interpret as time strings.
If stop < start, interpret stop as a width in seconds.
"""
import logging
import numpy as np
from astropy.time import Time
from snewpdag.dag import Node
from snewpdag.values import Hist1D

class NewHist1D(Node):
  def __init__(self, out_field, nbins, start, stop, **kwargs):
    self.out_field = out_field
    self.nbins = nbins
    self.start = start
    if isinstance(self.start, str):
      self.start = Time(self.start).to_value('unix', 'long')
    self.stop = stop
    if isinstance(self.stop, str):
      self.stop = Time(self.stop).to_value('unix', 'long')
    elif self.stop < self.start:
      self.stop = self.start + self.stop # interpret as width in seconds
    super().__init__(**kwargs)

  def alert(self, data):
    data[self.out_field] = Hist1D(self.nbins, self.start, self.stop)
    return data

