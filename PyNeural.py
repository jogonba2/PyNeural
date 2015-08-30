#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyNeural.py
#  
#  Copyright 2015 Overxflow13

from Exceptions import NonStableNeuralNetworkException
from VectorFuncts import vectorxvector,vectorplusvector,optovector
from ActivateFuncts import lineal,jump,sigmoid,hiperbolic_tangent,fast

def header():
	print """ ____  __ __  ____     ___  __ __  ____    ____  _     
|    \|  |  ||    \   /  _]|  |  ||    \  /    || |    
|  o  )  |  ||  _  | /  [_ |  |  ||  D  )|  o  || |    
|   _/|  ~  ||  |  ||    _]|  |  ||    / |     || |___ 
|  |  |___, ||  |  ||   [_ |  :  ||    \ |  _  ||     |
|  |  |     ||  |  ||     ||     ||  .  \|  |  ||     |
|__|  |____/ |__|__||_____| \__,_||__|\_||__|__||_____|
"""

def footer():
	print "By overxfl0w13.."
class Layer:
	
	def __init__(self,i,nUnits,theta): 
		self.i,self.nUnits,self.theta,self.units,self.s = i,nUnits,theta,[],0
	
	def _addUnit(self,unit):   self.units.append(unit); self.nUnits += 1
	def _addUnitWInit(self):   self.units.append(self.nUnits[-1]+1);     self.nUnits += 1
	def _setId(self,i):		   self.i = i
	def _setTheta(self,theta): self.theta = theta
	def _setS(self,s):		   self.s = s
	
	def _getUnit(self,i):    return self.units[i]
	def _getUnits(self):	 return self.units	
	def _getNUnits(self):    return self.nUnits
	def _getTheta(self):	 return self.theta
	def _getId(self):		 return self.i
	def _getS(self):		 return self.s
	
	def _checkLayer(self): return True if self.theta[0]==1 and self.nUnits==(len(self.theta)-1) else False
		
	
class PyNeural:
	
	def __init__(self,x,theta,nHiddenLayers,nHiddenUnits,outputUnits):
		self.x,self.nHiddenLayers,self.nHiddenUnits,self.outputUnits,self.hiddenLayers = x,nHiddenLayers,nHiddenUnits,outputUnits,[]
		self.theta = self.__initTheta(theta)
		self.__initializeHiddenLayers()
		err = self.__checkNeuralNetwork()
		if type(err)==type("str"): raise NonStableNeuralNetworkException(err)
		
	
	def _setX(self,x): 		       					   self.x = x
	def _setTheta(self,theta): 	   					   self.theta = theta
	def _setNHiddenLayers(self,nHiddenLayers):		   self.nHiddenLayers = nHiddenLayers
	def _setNHiddenUnits(self,nHiddenUnits):		   self.nHiddenUnits  = nHiddenUnits
	def _setOutputUnits(self,nOutputUnits):			   self.nOutputUnits  = nOutputUnits
	
	def _getX(self):								   return self.x
	def _getTheta(self):							   return self.theta
	def _getNHiddenLayers(self):					   return self.nHiddenLayers
	def _getNHiddenUnits(self):						   return self.nHiddenUnits
	def _getOutputUnits(self):						   return self.outputUnits
	def _getHiddenLayers(self):						   return self.hiddenLayers
	
	def __initTheta(self,theta): 
		if not theta:
			r = []
			for i in xrange(self.nHiddenLayers): r.append([1]+([0]*self.nHiddenUnits[i]))
			r.append([1]+([0]*self.outputUnits))
			# self._backwardPropagation() or _incrementalBackwardPropagation() or 
			# _momentumBackwardPropagation() or _bufferBackwardPropagation()
		else: r = theta
		return r

	def __checkNeuralNetwork(self):
		for layer in self.hiddenLayers:     
			if not layer._checkLayer():
				return "Layers are not stable"
		if len(self.theta)!=(self.nHiddenLayers+1):     return "Theta has not got the same length as nº of hidden layers"
		if self.outputUnits<=0:				            return "Output units is not valid"
		if not self.x:						            return "X is not valid"
		if not all(self.nHiddenUnits):				    return "Some layer has not valid number of hidden units"
		if len(self.nHiddenUnits)!=self.nHiddenLayers:  return "Number of hidden units specified does not match with the number of hidden layers"
		return True
		
	def __initializeHiddenLayers(self):
		for i in xrange(self.nHiddenLayers): self.hiddenLayers.append(Layer(i,self.nHiddenUnits[i],self.theta[i]))

	def _forwardPropagation(self): pass
	
	def _backwardPropagation(self): pass
	def _incrementalBackwardPropagation(self): pass
	def _momentumBackwardPropagation(self): pass
	def _bufferBackwardPropagation(self): pass
	
try:
	header()
	pNeural = PyNeural([1,2,3,4,5,6],[],2,[3,3],2)
	
except NonStableNeuralNetworkException as e:
	print e

finally:
	footer()
