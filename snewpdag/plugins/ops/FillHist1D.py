"""
FillHist1D:  a plugin which accumulates a histogram based on its configuration.
  Only notifies downstream plugins on a `report' action.

Constructor arguments:
  nbins: number of bins
  xlow: low edge of histogram
  xhigh: high edge of histogram
  in_field: input field specifier
  out_field: output field
  index_field (optional): index specifier for input

Output json:
  alert:  no output
  reset:  no output
  revoke:  no output
  report:  add (out_field)
"""
import sys
import logging
import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field
from snewpdag.values import Hist1D

class FillHist1D(Node):
  def __init__(self, nbins, xlow, xhigh, in_field, out_field, **kwargs):
    self.hist = Hist1D(nbins, xlow, xhigh)
    self.in_field = in_field
    self.out_field = out_field
    self.index_field = kwargs.pop('index_field', '')
    super().__init__(**kwargs)

  def clear(self):
    self.hist.clear()

  def alert(self, data):
    s = self.in_field if len(self.index_field) == 0 else list(self.in_field)
    if len(self.index_field) != 0:
      v, flag = fetch_field(data, self.index_field)
      if not flag:
        logging.error('{}: index field {} not found'.format(self.name, self.index_field))
        return False
      s.append(v)
    v, flag = fetch_field(data, s)
    if not flag:
      logging.error('{}: field {} not found'.format(self.name, s))
      return False
    self.hist.fill(v)
    return False # don't forward an alert

  def reset(self, data):
    return False

  def revoke(self, data):
    return False

  def report(self, data):
    data[self.out_field] = self.hist.copy()
    return True

