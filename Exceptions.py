#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Exceptions.py
#  
#  Copyright 2015 Overxflow13


class NonStableNeuralNetworkException(Exception):
	
	def __init__(self,value): self.value = "[-] NonStableNeuralNetworkException: Parameters of neural network aren't stable: "+str(value)
		
	def __str__(self): return self.value

		
class NonVectorSameLengthException(Exception):	
	def __init__(self): self.value = "[-] NonVectorSameLengthException: Vectors haven't got the same length"
		
	def __str__(self): return self.value
