#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyNeural.py
#  
#  Copyright 2015 Overxflow13

from Exceptions import NonStableNeuralNetworkException
from ActivateFuncts import lineal,jump,sigmoid,hiperbolic_tangent,fast

def header():
	print """ ____  __ __  ____     ___  __ __  ____    ____  _     
|    \|  |  ||    \   /  _]|  |  ||    \  /    || |    
|  o  )  |  ||  _  | /  [_ |  |  ||  D  )|  o  || |    
|   _/|  ~  ||  |  ||    _]|  |  ||    / |     || |___ 
|  |  |___, ||  |  ||   [_ |  :  ||    \ |  _  ||     |
|  |  |     ||  |  ||     ||     ||  .  \|  |  ||     |
|__|  |____/ |__|__||_____| \__,_||__|\_||__|__||_____|\n\n
"""

def footer():
	print "\n\n---------------\nBy overxfl0w13.\n---------------\n"
	
class Layer:
	
	def __init__(self,i,nUnits,theta): 
		self.i,self.nUnits,self.theta,self.s = i,nUnits,theta,[]
	
	def _setId(self,i):		   self.i = i
	def _setTheta(self,theta): self.theta = theta
	def _setS(self,s):		   self.s = s
			
	def _getNUnits(self):    return self.nUnits
	def _getTheta(self):	 return self.theta
	def _getId(self):		 return self.i
	def _getS(self):		 return self.s
		
	
class PyNeural:
	
	def __init__(self,x,theta,nHiddenLayers,nHiddenUnits,outputUnits,fActivate):
		self.x,self.nHiddenLayers,self.nHiddenUnits,self.outputUnits,self.hiddenLayers,self.theta,self.fActivate = self.__normalizeX(x),nHiddenLayers,nHiddenUnits,outputUnits,[],theta,fActivate
		self.__initializeLayers()
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
	
	def __normalizeX(self,x): x.insert(0,1) ; return x		

	def __checkNeuralNetwork(self):
		if not self.theta:								return "Theta is not modified correctly"
		if len(self.theta)!=(self.nHiddenLayers+1):     return "Theta has not got the same length as nº of hidden layers"
		if self.outputUnits<=0:				            return "Output units is not valid"
		if not self.x:						            return "X is not valid"
		if not all(self.nHiddenUnits):				    return "Some layer has not valid number of hidden units"
		if len(self.nHiddenUnits)!=self.nHiddenLayers:  return "Number of hidden units specified does not match with the number of hidden layers"
		return True
		
	def __initializeLayers(self):
		c = 0
		for i in xrange(self.nHiddenLayers): 
			self.hiddenLayers.append(Layer(i,self.nHiddenUnits[i],self.theta[i])) ; c += 1
		self.hiddenLayers.append(Layer(c,self.outputUnits,self.theta[c]))

	def __computeSLayer(self,layer,s=None):
		thetaLayer = layer._getTheta()
		nUnits  = layer._getNUnits()
		idLayer = layer._getId()
		sLayer  = []
		for i in xrange(nUnits):
			r = 0
			if idLayer==0: s = self.x
			for j in xrange(len(s)):
				r += thetaLayer[i][j]
			sLayer.append(self.fActivate(r))				
		return sLayer
			
		
	def _forwardPropagation(self):
		c = 0
		for i in xrange(self.nHiddenLayers+1):
			layer  = self.hiddenLayers[i]
			if i==0: s = None
			else:    s = self.hiddenLayers[i-1]._getS()
			sLayer = self.__computeSLayer(layer,s)
			layer._setS(sLayer)
			c += 1			
			print "Layer",i,"results:",str(layer._getS())
				
	
	def _backwardPropagation(self): pass
	def _incrementalBackwardPropagation(self): pass
	def _momentumBackwardPropagation(self): pass
	def _bufferBackwardPropagation(self): pass
	
	def __str__(self): pass
	
try:
	header()
	theta = [
		[
			[1,0.5,0.2,0.8],[1,0.3,0.7,0.4],[1,0.3,0.7,0.4]
		],
		[
			[1,0.1,0.3,0.9]
		]
	]
	pNeural = PyNeural([5.2,4.9,3.5],theta,1,[3],1,sigmoid)
	
	pNeural._forwardPropagation()
	
except NonStableNeuralNetworkException as e:
	print e

finally:
	footer()
