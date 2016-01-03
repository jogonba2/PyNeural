#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import e

### Funciones de activacion ###

def lineal(z): return z

def step(z): return 1.0 if z>0 else -1.0

def ramp(z):
	if z>=1: return 1.0
	elif -1<z<1: return z
	elif z<=-1: return -1.0
	
def sigmoid(z):
	return 1.0/(1.0+(e**(-z)))

def tanh(z): return ((e**z)-(e**-z))/((e**z)+(e**-z))

def fast(z): return z/(1+abs(z))

################################

### Derivadas de las funciones de activacion ###

def derivative_lineal(z):  return 1.0
def derivative_sigmoid(z): return sigmoid(z)*(1-sigmoid(z))
def derivative_tanh(z):    return 1.0-(tanh(z)**2)
def derivative_fast(z):    return 1.0/((1+abs(z))**2)
