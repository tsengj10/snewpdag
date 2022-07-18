"""
Unit tests for dag library methods
"""
import unittest
import numpy as np
from snewpdag.dag.lib import *

class TestLib(unittest.TestCase):

  def test_convert(self):
    t = time_tuple_from_float(3.5)
    self.assertEqual(t[0], 3)
    self.assertEqual(t[1], 500000000)
    t = time_tuple_from_float([4.5, 2.25])
    self.assertEqual(t[0][0], 4)
    self.assertEqual(t[0][1], 500000000)
    self.assertEqual(t[1][0], 2)
    self.assertEqual(t[1][1], 250000000)
    t = time_tuple_from_float(-3.5)
    self.assertEqual(t[0], -4)
    self.assertEqual(t[1], 500000000)
    t = time_tuple_from_float([-4.5, -0.25])
    self.assertEqual(t[0][0], -5)
    self.assertEqual(t[0][1], 500000000)
    self.assertEqual(t[1][0], -1)
    self.assertEqual(t[1][1], 750000000)
    t = time_tuple_from_offset(8250000005)
    self.assertEqual(t[0], 8)
    self.assertEqual(t[1], 250000005)
    t = time_tuple_from_offset([1234567890, 2345678901])
    self.assertEqual(t[0][0], 1)
    self.assertEqual(t[0][1], 234567890)
    self.assertEqual(t[1][0], 2)
    self.assertEqual(t[1][1], 345678901)
    t = offset_from_time_tuple((3,5))
    self.assertEqual(t, 3000000005)
    t = offset_from_time_tuple([(3,5), (-1,5)])
    self.assertEqual(t[0], 3000000005)
    self.assertEqual(t[1], -999999995)

  def test_single(self):
    g = ns_per_second
    ti = (5, 40)
    to = normalize_time(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], 40)
    ti = (5, -40)
    to = normalize_time(ti)
    self.assertEqual(to[0], 4)
    self.assertEqual(to[1], g-40)
    ti = (-5, 40)
    to = normalize_time(ti)
    self.assertEqual(to[0], -5)
    self.assertEqual(to[1], 40)
    ti = (-5, -40)
    to = normalize_time(ti)
    self.assertEqual(to[0], -6)
    self.assertEqual(to[1], g-40)
    ti = (5, g+40)
    to = normalize_time(ti)
    self.assertEqual(to[0], 6)
    self.assertEqual(to[1], 40)
    ti = (5, g-40)
    to = normalize_time(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], g-40)
    ti = (5, 0)
    to = normalize_time(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], 0)
    ti = (5, g)
    to = normalize_time(ti)
    self.assertEqual(to[0], 6)
    self.assertEqual(to[1], 0)
    ti = (-5, 0)
    to = normalize_time(ti)
    self.assertEqual(to[0], -5)
    self.assertEqual(to[1], 0)
    ti = (-5, g)
    to = normalize_time(ti)
    self.assertEqual(to[0], -4)
    self.assertEqual(to[1], 0)
    ti = (-5, -g)
    to = normalize_time(ti)
    self.assertEqual(to[0], -6)
    self.assertEqual(to[1], 0)

  def test_multi(self):
    g = ns_per_second
    ti = [ (5, 40), (5, -40), (-5, 40), (-5, -40), (5, g+40), (5, g-40) ]
    to = normalize_time(ti)
    self.assertEqual(to[0,0], 5)
    self.assertEqual(to[1,0], 4)
    self.assertEqual(to[2,0], -5)
    self.assertEqual(to[3,0], -6)
    self.assertEqual(to[4,0], 6)
    self.assertEqual(to[5,0], 5)
    self.assertEqual(to[0,1], 40)
    self.assertEqual(to[1,1], g-40)
    self.assertEqual(to[2,1], 40)
    self.assertEqual(to[3,1], g-40)
    self.assertEqual(to[4,1], 40)
    self.assertEqual(to[5,1], g-40)

  def test_norm_dt(self):
    g = ns_per_second
    ti = (5, 40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], 40)
    ti = (5, -40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 4)
    self.assertEqual(to[1], g-40)
    ti = (-5, 40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], -4)
    self.assertEqual(to[1], -g+40)
    ti = (-5, -40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], -5)
    self.assertEqual(to[1], -40)
    ti = (5, g+40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 6)
    self.assertEqual(to[1], 40)
    ti = (5, g-40)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], g-40)
    ti = (5, 0)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 5)
    self.assertEqual(to[1], 0)
    ti = (5, g)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], 6)
    self.assertEqual(to[1], 0)
    ti = (-5, 0)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], -5)
    self.assertEqual(to[1], 0)
    ti = (-5, g)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], -4)
    self.assertEqual(to[1], 0)
    ti = (-5, -g)
    to = normalize_time_difference(ti)
    self.assertEqual(to[0], -6)
    self.assertEqual(to[1], 0)

  def test_subtract(self):
    g = ns_per_second
    a = (5, 40)
    b = (5, 30)
    c = subtract_time(a, b)
    self.assertEqual(c[0], 0)
    self.assertEqual(c[1], 10)
    c = subtract_time(b, a)
    self.assertEqual(c[0], 0)
    self.assertEqual(c[1], -10)

  def test_fetch_field(self):
    data1 = { 'f10': 10, 'f11': 11 }
    data2 = { 'f20': data1, 'f21': 21 }
    data3 = { 'f30': 30, 'f31': data2 }
    data4 = { 'f40': 40, 'f41': data3 }
    v, flag = fetch_field(data4, 'f40')
    self.assertTrue(flag)
    self.assertEqual(v, 40)
    v, flag = fetch_field(data4, ('f40',))
    self.assertTrue(flag)
    self.assertEqual(v, 40)
    v, flag = fetch_field(data4, ('f41','f30',))
    self.assertTrue(flag)
    self.assertEqual(v, 30)
    v, flag = fetch_field(data4, ['f41','f31','f21'])
    self.assertTrue(flag)
    self.assertEqual(v, 21)
    v, flag = fetch_field(data4, ['f41','f31','f20','f11'])
    self.assertTrue(flag)
    self.assertEqual(v, 11)
    v, flag = fetch_field(data4, ('f41','f31','f20',))
    self.assertTrue(flag)
    self.assertEqual(v, data1)
    v, flag = fetch_field(data4, ('f42',))
    self.assertFalse(flag)
    self.assertEqual(v, None)
    v, flag = fetch_field(data4, ('f41','f32'))
    self.assertFalse(flag)
    self.assertEqual(v, None)
    v, flag = fetch_field(data4, ('f41','f30','f20',))
    self.assertFalse(flag)
    self.assertEqual(v, None)

  def test_fetch_array(self):
    data0 = { 'f00': 55, 'f01': [ 7.5, 8.5 ] }
    data1 = { 'f10': [1.5, 2.5, data0], 'f11': np.array([4.5, 5.5, 6.5]) }
    data2 = { 'f20': data1, 'f21': 21 }
    v, flag = fetch_field(data2, ('f20','f10',1))
    self.assertTrue(flag)
    self.assertEqual(v, 2.5)
    v, flag = fetch_field(data2, ('f20','f10',5))
    self.assertFalse(flag)
    v, flag = fetch_field(data2, ('f20','f11',2))
    self.assertTrue(flag)
    self.assertEqual(v, 6.5)
    v, flag = fetch_field(data2, ('f20','f11',3))
    self.assertFalse(flag)
    v, flag = fetch_field(data2, ('f20','f10', 2, 'f00'))
    self.assertTrue(flag)
    self.assertEqual(v, 55)
    v, flag = fetch_field(data2, ('f20','f10', 2, 'f01', 0))
    self.assertTrue(flag)
    self.assertEqual(v, 7.5)

