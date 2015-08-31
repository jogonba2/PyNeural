#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyNeural.py
#  
#  Copyright 2015 Overxflow13

from Exceptions import NonStableNeuralNetworkException
from ActivateFuncts import lineal,jump,sigmoid,hiperbolic_tangent,fast

def header():
	print """ -------------------------------------------------------\n
 ____  __ __  ____     ___  __ __  ____    ____  _     
|    \|  |  ||    \   /  _]|  |  ||    \  /    || |    
|  o  )  |  ||  _  | /  [_ |  |  ||  D  )|  o  || |    
|   _/|  ~  ||  |  ||    _]|  |  ||    / |     || |___ 
|  |  |___, ||  |  ||   [_ |  :  ||    \ |  _  ||     |
|  |  |     ||  |  ||     ||     ||  .  \|  |  ||     |
|__|  |____/ |__|__||_____| \__,_||__|\_||__|__||_____|
\n-------------------------------------------------------\n\n
"""

def footer():
	print "\n\n---------------\nBy overxfl0w13.\n---------------\n"

class Layer:
	
	def __init__(self,i,nUnits,theta): self.i,self.nUnits,self.theta,self.s = i,nUnits,theta,[]
	
	def _setId(self,i):		   self.i = i
	def _setTheta(self,theta): self.theta = theta
	def _setS(self,s):		   self.s = s
			
	def _getNUnits(self):    return self.nUnits
	def _getTheta(self):	 return self.theta
	def _getId(self):		 return self.i
	def _getS(self):		 return self.s
		
	
class PyNeural:
	
	def __init__(self,x,theta,nHiddenLayers,nHiddenUnits,outputUnits,fActivate):
		self.x,self.nHiddenLayers,self.nHiddenUnits,self.outputUnits,self.hiddenLayers,self.theta,self.fActivate = x,nHiddenLayers,nHiddenUnits,outputUnits,[],theta,fActivate
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

	def __computeSLayer(self,layer,s):
		thetaLayer,nUnits,idLayer,sLayer = layer._getTheta(),layer._getNUnits(),layer._getId(),[]
		if VERBOSE: print "-----\nLayer",idLayer,"\n-----"
		for i in xrange(nUnits):
			r = 0	
			if VERBOSE: print "\t----\n\tUnit",i,"\n\t----"	
			for j in xrange(len(s)):
				if VERBOSE: print "\t\tR: ",thetaLayer[i][j],"*",s[j]
				r += thetaLayer[i][j]*s[j]
			if VERBOSE: print "\t\tSum:",r,"\n\t\tFActivate(Sum) =",self.fActivate(r)
			sLayer.append(self.fActivate(r))				
		return sLayer
			
		
	def _forwardPropagation(self):
		c = 0
		for i in xrange(self.nHiddenLayers+1):
			layer  = self.hiddenLayers[i]
			if i==0: s = self.x
			else:    s = self.hiddenLayers[i-1]._getS()
			sLayer = self.__computeSLayer(layer,s)
			layer._setS(sLayer)
			c += 1						
	
	""" IOI _forwardPropagation was executed """
	""" It assumes 1 output unit per class in output layer """
	def _classify(self):
		outputLayer = self.hiddenLayers[len(self.hiddenLayers)-1]
		m,im,r = float("-inf"),-1,outputLayer._getS()
		for i in xrange(len(r)):
			if r[i]>m: m,im = r[i],i
		print "\n\n---------------------\nClassification result\n---------------------\n\n\t-> Class",im
	
	""" IOI _forwardPropagation was executed """	
	""" Regression of R^d -> R^d' with d = len(InputLayer) and d' = len(outputLayer)"""	
	def _regression(self):	print "\n\n-----------------\nRegression result\n-----------------\n\n",str(self.hiddenLayers[len(self.hiddenLayers)-1]._getS())
		
	def _backwardPropagation(self): pass
	def _incrementalBackwardPropagation(self): pass
	def _momentumBackwardPropagation(self): pass
	def _bufferBackwardPropagation(self): pass
	
	
try:
	header()
	VERBOSE,CLASSIFY,REGRESSION = 1,0,0
	
	print "------------ Example ------------ \n\n"
	
	# Example of: https://gyazo.com/27e1802a18be451bef187dc1cc208b24 # 
	
	x     = [5.2,4.9,3.5,2.7,1.3]
	theta = [
		[
			[1,0.5,0.2,0.8,0.4],[1,0.3,0.7,0.4,0.3],[1,0.3,0.7,0.4,0.7],
		],
		[
			[1,0.5,0.2,0.8],[1,0.3,0.7,0.4],[1,0.3,0.7,0.4]
		],
		[
			[1,0.5,0.2],[1,0.3,0.7],[1,0.3,0.7],[1,0.25,0.37]
		]
	]
	nHiddenLayers  = 2
	nUnitsPerLayer = [3,3]
	outputUnits    = 4
	fActivate      = sigmoid
	pNeural = PyNeural(x,theta,nHiddenLayers,nUnitsPerLayer,outputUnits,sigmoid)
	pNeural._forwardPropagation()
	pNeural._classify()
	pNeural._regression()
	
	print "\n\n ------------ Another example ------------ \n\n"
	
	# Example of: https://gyazo.com/2a011b7b0c42c6a0cd73970c3bacc9a8 #
	
	x     = [1.25,3.21,4.56,2.35,7.53]
	theta = [
		[
			[1,0.5,0.2,0.8,0.4],[1,0.3,0.7,0.4,0.3],[1,0.3,0.7,0.4,0.7],
		],
		[
			[1,0.5,0.2],[1,0.35,0.45],[1,0.28,0.78]
		]
	]
	nHiddenLayers  = 1
	nUnitsPerLayer = [3]
	outputUnits    = 3
	fActivate      = sigmoid

	pNeural = PyNeural(x,theta,nHiddenLayers,nUnitsPerLayer,outputUnits,sigmoid)
	pNeural._forwardPropagation()
	pNeural._classify()
	pNeural._regression()
	
except Exception as e: print e

finally: footer()
