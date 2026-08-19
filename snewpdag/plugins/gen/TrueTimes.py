"""
TrueTimes - generate true arrival times

Arguments:
  detector_location: filename of detector database for DetectorDB
  detectors: list of detectors for which to generate burst times
  ra: right ascension (degrees)
  dec: declination (degrees)
  nside: healpix nside
  iside: input nside
  ipix: input healpix pixel number
    (iside,pixel) takes precedence over (ra,dec)
  time: time string, e.g., '2021-11-01 05:22:36.328'
  epoch_base (optional): starting time for epoch, float value or field specifier
    (string or tuple)

  If (iside,ipix) specified, the (ra,dec) will be calculated based on
  (iside,ipix).  The true pixel for (ra,dec) will then be found for
  the given nside value, which may be different from iside.

Input:
  [epoch_base]: float value of starting time for epoch, in unix epoch

Output:
  truth/true_sn_ra: right ascension (radians)
  truth/true_sn_dec: declination (radians)
  truth/true_sn_nside: true pixel nside
  truth/true_sn_pixel: true pixel number
  truth/dets/<det_id>/true_t: arrival time, (float) seconds in snewpdag time
"""
import logging
import numbers
import numpy as np
import healpy as hp
from astropy.time import Time
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord, CartesianRepresentation

from snewpdag.dag import Node, Detector, DetectorDB
from snewpdag.dag.lib import fetch_field

class TrueTimes(Node):
  def __init__(self, detector_location, detectors, time, **kwargs):
    self.db = DetectorDB(detector_location)
    self.time = Time(time)
    self.ra_d = kwargs.pop('ra', -60.0)
    self.dec_d = kwargs.pop('dec', -30.0)
    self.nside = kwargs.pop('nside', 8)
    self.iside = kwargs.pop('iside', 8)
    self.pixel = kwargs.pop('ipix', -1)
    self.time_unix = self.time.to_value('unix', 'long') # float, unix epoch
    self.epoch_base = kwargs.pop('epoch_base', 0.0)
    self.frame = kwargs.pop('frame', 'icrs')

    if self.pixel >= 0:
      # if ipix selected, choose angular position of pixel in iside map
      (self.ra_d, self.dec_d) = hp.pix2ang(self.iside, self.pixel, \
                                           nest=True, lonlat=True)

    self.ra = np.radians(self.ra_d)
    self.dec = np.radians(self.dec_d)

    # select pixel in output nside
    self.pixel = hp.ang2pix(self.nside, self.ra_d, self.dec_d, \
                            nest=True, lonlat=True)

    logging.info('TrueTimes (ra,dec) = ({},{}) deg'.format(self.ra_d, self.dec_d))
    logging.info('TrueTimes (ra,dec) = ({},{})'.format(self.ra, self.dec))
    logging.info('TrueTimes (nside,ipix) = ({},{})'.format(self.nside, self.pixel))

    if not isinstance(self.epoch_base, (numbers.Number, str, list, tuple)):
      logging.error('TrueTimes.__init__: unrecognized epoch_base {}. Set to 0.'.format(self.epoch_base))
      self.epoch_base = 0.0

    # use ra,dec in degrees, not self.ra,self.dec which are in radians
    sc = SkyCoord(ra=self.ra_d, dec=self.dec_d, unit=u.deg, frame=self.frame, \
                  representation_type='unitspherical', obstime=self.time)
    gc = sc.transform_to(GCRS)
    d = gc.represent_as(CartesianRepresentation)
    self.snr = np.array( [ d.x, d.y, d.z ] ) # should be unit length!
    logging.info('ra(lon) = {}, dec(lat) = {}'.format(self.ra, self.dec))
    logging.info('SN location = {}'.format(self.snr))
    self.dets = set(detectors) # detector names
    super().__init__(**kwargs)

  def alert(self, data):
    # record truth information
    if 'truth' not in data:
      data['truth'] = {}
    data['truth']['true_sn_ra'] = self.ra # radians
    data['truth']['true_sn_dec'] = self.dec # radians
    data['truth']['true_sn_nside'] = self.nside
    data['truth']['true_sn_pixel'] = self.pixel

    # epoch base
    if isinstance(self.epoch_base, numbers.Number):
      t0 = self.epoch_base
    elif isinstance(self.epoch_base, (str, list, tuple)):
      t0 = fetch_field(data, self.epoch_base)
    time_base = self.time_unix - t0

    # generate true times for each detector.
    # given time is when wavefront arrives at Earth origin.
    ts = {}
    for dname in self.dets:
      #c = 3.0e8 # m/s
      det = self.db.get(dname)
      pos = det.get_xyz(self.time) # detector in GCRS at given time
      logging.info('pos[{}] = {}'.format(dname, pos))
      logging.info('  sn pos = {}'.format(self.snr))
      dt = - np.dot(det.get_xyz(self.time), self.snr) / const.c # intersect
      t1 = time_base + dt.to(u.s).value
      logging.info('  true_t = {}'.format(t1))
      ts[dname] = {
                    'true_t': t1, # s in snewpdag time
                  }

    data['truth']['dets'] = ts
    return data

  def report(self, data):
    # record truth information
    if 'truth' not in data:
      data['truth'] = {}
    data['truth']['true_sn_ra'] = self.ra # radians
    data['truth']['true_sn_dec'] = self.dec # radians
    data['truth']['true_sn_nside'] = self.nside
    data['truth']['true_sn_pixel'] = self.pixel
    return data

