#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  VectorFuncts.py
#  
#  Copyright 2015 Overxflow13

from Exceptions import NonVectorSameLengthException

""" SII |x|==|y| """
def vectorxvector(x,y): 
	s = 0
	if len(x)!=len(y): raise NonVectorSameLengthException
	for i in xrange(len(x)): s += x[i]*y[i]
	return s

""" SII |x|==|y| """
def vectorplusvector(x,y): 
	if len(x)!=len(y): raise NonVectorSameLengthException
	return [x[i]+y[i] for i in xrange(len(x))] 

""" SII |x|==|y| """
def vectordiffvector(x,y): 
	if len(x)!=len(y): raise NonVectorSameLengthException
	return [x[i]-y[i] for i in xrange(len(x))]
	
def optovector(x,op): return map(op,x)

def vectorxconstant(x,a): return [i*a for i in x]

