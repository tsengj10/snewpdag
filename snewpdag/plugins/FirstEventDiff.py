"""
FirstEventDiff - first event method, with yield bias correction

Arguments:
  in_series1_field
  in_series2_field
  out_field:  output field, dictionary fields:
      delta = delta based on first events of series
      exp_delta = expectation value of delta if there was no lag (bias)
      diff = delta - exp_delta (yield bias-corrected delta)
      sigma_diff = expected uncertainty on diff
  true_lag = true lag value (optional)
  true_t1 = true arrival time for detector 1 (optional)
  true_t2 = true arrival time for detector 2 (optional)
"""
import logging
import numbers
import numpy as np

from snewpdag.dag import Node
from snewpdag.dag.lib import fetch_field, store_field

class FirstEventDiff(Node):
  def __init__(self, in_series1_field, in_series2_field, out_field, **kwargs):
    self.in_series1_field = in_series1_field
    self.in_series2_field = in_series2_field
    self.out_field = out_field
    self.in_ref_field = kwargs.pop('in_ref_field', in_series1_field)
    self.true_lag = kwargs.pop('true_lag', 0.0)
    self.true_t1 = kwargs.pop('true_t1', 0.0)
    self.true_t2 = kwargs.pop('true_t2', 0.0)
    self.true_tr = kwargs.pop('true_tr', self.true_t1)
    self.out_key = kwargs.pop('out_key', ('D1','D2')) # usually names the detectors
    self.sigma_fudge = kwargs.pop('sigma_fudge', 1.0) # scale up sigma
    super().__init__(**kwargs)

  def alert(self, data):
    tsr1, valid = fetch_field(data, self.in_series1_field) # TimeSeries
    if not valid:
      return False
    tsr2, valid = fetch_field(data, self.in_series2_field) # TimeSeries
    if not valid:
      return False
    tsref, valid = fetch_field(data, self.in_ref_field) # TimeSeries
    if not valid:
      return False

    # difference between first times
    tf1 = np.min(tsr1.times)
    tf2 = np.min(tsr2.times)
    dtf = tf1 - tf2

    # subtract off one of the first times
    #base = tf1 if tf1 < tf2 else tf2
    base = np.min([ tf1, tf2, np.min(tsref.times) ])
    ts1 = tsr1.times - base
    ts2 = tsr2.times - base
    tsr = tsref.times - base
    #tf1 = tf1 - base
    #tf2 = tf2 - base

    # expected value of delta
    # note that aside from alpha, this only depends on first series
    alpha1 = len(ts1) / len(tsr)
    alpha2 = len(ts2) / len(tsr)
    s1 = np.sort(ts1)
    s2 = np.sort(ts2)
    sr = np.sort(tsr)
    ikr = np.arange(1.0, len(sr) + 1.0)
    e1 = np.exp(-alpha1*ikr)
    et1 = np.sum(e1 * sr) / np.sum(e1) # exp val of t1
    et1sq = np.sum(e1 * sr * sr) / np.sum(e1)
    e1a = np.exp(-alpha2*ikr)
    et1a = np.sum(e1a * sr) / np.sum(e1a) # exp val of t1 with different yield
    et1asq = np.sum(e1a * sr * sr) / np.sum(e1a)

    ik1 = np.arange(1.0, len(s1) + 1.0)
    e1_noa = np.exp(-ik1)
    et1_noa = np.sum(e1_noa * s1) / np.sum(e1_noa)
    et1sq_noa = np.sum(e1_noa * s1 * s1) / np.sum(e1_noa)

    ik2 = np.arange(1.0, len(s2) + 1.0)
    e2 = np.exp(-ik2)
    et2 = np.sum(e2 * s2) / np.sum(e2) # exp val of t1 of second experiment
    et2sq = np.sum(e2 * s2 * s2) / np.sum(e2)

    dte = et1 - et1a # bias due to alpha (different yields)

    # deviation (diff - expected diff)
    dev = dtf - dte
    logging.debug('{}: dtf = {}, dte = {}, dev = {}'.format(self.name, dtf, dte, dev))
    logging.debug('{}: s1({}) = {}'.format(self.name, len(s1), s1[:5]))
    logging.debug('{}: s2({}) = {}'.format(self.name, len(s2), s2[:5]))
    logging.debug('{}: sr({}) = {}'.format(self.name, len(sr), sr[:5]))

    # uncertainty estimate
    #sigma2 = et1sq + et2sq - et1*et1 - et2*et2
    var1 = et1sq - et1*et1
    var2 = et2sq - et2*et2
    var1a = et1asq - et1a*et1a
    var1_noa = et1sq_noa - et1_noa*et1_noa
    # choose the larger variance between 2 and 1a
    if var1_noa > var1: # larger variance between 1 and 1_noa
      var1 = var1_noa
    if var1a > var2:
      sigma2 = var1 + var1a
      var2 = var1a
    else:
      sigma2 = var1 + var2
    rms = np.sqrt(sigma2)
    rms_fudge = rms * self.sigma_fudge
    #logging.debug('{}: et1sq = {}, et2sq = {}, et1 = {}, et2 = {}'.format(self.name, et1sq, et2sq, et1, et2))
    logging.debug('{}: et1sq = {}, et2sq = {}, et1asq = {}'.format(self.name, et1sq, et2sq, et1asq))
    logging.debug('{}: et1 = {}, et2 = {}, et1a = {}'.format(self.name, et1, et2, et1a))

    # pull
    dt_true = dev - self.true_lag
    pull = (dev - self.true_lag) / rms
    pull_fudge = (dev - self.true_lag) / rms_fudge

    dtf_true = dtf - self.true_lag # uncorrected diff - true

    true_t1 = 0.0
    true_t2 = 0.0
    true_tref = 0.0
    if isinstance(self.true_t1, numbers.Number):
      true_t1 = self.true_t1
    elif isinstance(self.true_t1, (str, list, tuple)):
      true_t1, valid = fetch_field(data, self.true_t1)
    if isinstance(self.true_t2, numbers.Number):
      true_t2 = self.true_t2
    elif isinstance(self.true_t2, (str, list, tuple)):
      true_t2, valid = fetch_field(data, self.true_t2)
    if isinstance(self.true_tr, numbers.Number):
      true_tref = self.true_tr
    elif isinstance(self.true_tr, (str, list, tuple)):
      true_tref, valid = fetch_field(data, self.true_tr)
    logging.debug('{}: tf1 = {}, tf2 = {}'.format(self.name, tf1, tf2))
    logging.debug('{}: true_t1 = {}, true_t2 = {}, true_tref = {}'.format(self.name, true_t1, true_t2, true_tref))

    # assemble output dictionary
    #d = { 'delta': dtf, 'exp_delta': dte, 'diff': dev, 'sigma_diff': rms, 'pull': pull }
    #logging.debug('{}:  diff = {}, sigma_diff = {}, pull = {}'.format(self.name, dev, rms, pull))
    #store_field(data, self.out_field, d)
    d, exists = fetch_field(data, self.out_field) # to append
    if exists:
      dts = d.copy() # shallow copy of the dts dictionary so we can add to it
    else:
      dts = {}
    dts[self.out_key] = {
                          'delta': dtf, # not needed by DiffPointing
                          'exp_delta': dte, # not needed by DiffPointing
                          'dt_true': dt_true, # not needed by DiffPointing
                          'dtf_true': dtf_true, # not needed by DiffPointing
                          'pull': pull, # not needed by DiffPointing
                          'var1': et1sq - et1*et1, # not needed by DiffPointing
                          'var2': et2sq - et2*et2, # not needed by DiffPointing
                          'var1a': et1asq - et1a*et1a, # not needed by DiffPointing
                          'var1_noa': et1sq_noa - et1_noa*et1_noa, # not needed by DiffPointing
                          'rms': rms, # not needed by DiffPointing
                          'pull_fudge': pull_fudge, # not for DP, incl fudge
                          'rms_fudge': rms_fudge, # not for DP, incl fudge
                          'dt': dev,
                          #'t1': tf1 - et1, # individual bias corrected estimate of first event time
                          #'t2': tf2 - et1a,
                          'ts1_noa': et1_noa + base - true_t1, # not for DP
                          'ts2_noa': et2 + base - true_t2,
                          'ts1': et1 + base - true_tref,
                          'ts2': et1a + base - true_tref,
                          'bias': 0.0,
                          'var': sigma2 * self.sigma_fudge * self.sigma_fudge,
                          'dsig1': np.sqrt(var1) * self.sigma_fudge,
                          'dsig2': - np.sqrt(var2) * self.sigma_fudge,
                        }
    store_field(data, self.out_field, dts)

    """
    # debug:  if pull > 5
    if pull > 5.0:
      logging.info('{}:--debug for pull {}'.format(self.name, pull))
      logging.info('{}:  dts = {}'.format(self.name, dts[self.out_key]))
      logging.info('{}:  tf1 = {}, tf2 = {}, dtf = {}, base = {}'.format(self.name, tf1, tf2, dtf, base))
      logging.info('{}:  s1({}) = {}'.format(self.name, len(s1), s1[:10]))
      logging.info('{}:  s2({}) = {}'.format(self.name, len(s2), s2[:10]))
      logging.info('{}:  length of tsr1,tsr2 = {}, {}'.format(self.name, len(tsr1.times), len(tsr2.times)))
      sr1 = np.sort(tsr1.times)
      sr2 = np.sort(tsr2.times)
      for j in range(10):
        logging.info('{}:  [{}] = {} {}'.format(self.name, j, sr1[j], sr2[j]))
      logging.info('{}:  et1 = {}, et2 = {}, dte = {}'.format(self.name, et1, et2, dte))
      logging.info('{}:  et1sq = {}, et2sq = {}, et1asq = {}'.format(self.name, et1sq, et2sq, et1asq))
      logging.info('{}:  et1 = {}, et2 = {}, et1a = {}'.format(self.name, et1, et2, et1a))
      logging.info('{}:  truth/dets = {}'.format(self.name, data['truth']['dets']))
      # find index of minimum element
      i2 = 0
      m2 = tsr2.times[0]
      for j in range(len(tsr2.times)):
        if tsr2.times[j] < m2:
          m2 = tsr2.times[j]
          i2 = j
      logging.info('{}:  index of min time in tsr2 is {} (value {})'.format(self.name, i2, m2))
      logging.info('{}:  test argmin tsr2 is {}'.format(self.name, np.argmin(tsr2.times)))
    """

    return True

