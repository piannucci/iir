#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name='IIR',
      version='1.0.0',
      description='Infinite Impulse Response Filter Utilities',
      author='Peter Iannucci',
      author_email='iannucci@mit.edu',
      url='',
      packages=['iir'],
      ext_modules=[Extension('mkfilter', ['src/mkfilter.cc'])],
      )
