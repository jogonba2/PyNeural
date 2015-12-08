#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ActivateFuncts.py
#  
#  Author: Overxflow13

from math import e

def lineal(z): return z

def jump(z):   return 1 if z>0 else -1

def sigmoid(z):  return 1.0/(1+(e**-z))

def hiperbolic_tangent(z): return ((e**z)-(e**-z))/((e**z)+(e**-z))

def fast(z): return 1.0/(1+abs(z))



