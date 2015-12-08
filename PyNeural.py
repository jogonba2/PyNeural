#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyNeural.py
#  
#  Author: Overxflow13

import numpy as np
from ActivationFuncts import lineal,jump,sigmoid,hiperbolic_tangent,fast

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


def footer(): print "\n\n---------------\nBy overxfl0w13.\n---------------\n"
	
def forward_propagation(x,theta,factivation):
	s = x
	s_layers = []
	for layer in xrange(len(theta)):
		aux = [1.0]
		for j in xrange(len(theta[layer])): aux.append(factivation[layer](sum(theta[layer][j]*s)))
		s = aux
		s_layers.append(s)
	return s_layers

def classify(x,theta,factivation):
	s_layers = forward_propagation(x,theta,factivation)
	output_layer = list(s_layers[-1][1:])
	return output_layer.index(max(output_layer))

def regression(x,theta,factivarion): return forward_propagation(x,theta,factivation)[-1][1:]

def back_propagation(S,theta,hidden_layers,neurons_by_layer,factivation,ro=1,max_iterations=250): 
	
	delta,theta_ant,fixed,it      = [],[],False,0

	for i in xrange(hidden_layers+1):
		aux = []
		for j in xrange(neurons_by_layer[i]): aux.append(np.copy(theta[i][j]))
		theta_ant.append(aux)
		
	while fixed==False and it<max_iterations:
		for (x,t) in S: 
			s_layers = forward_propagation(x,theta,factivation)
			s_output_layer = s_layers[hidden_layers][1:]
			for l in xrange(hidden_layers,-1,-1):
				aux = []
				for i in xrange(0,neurons_by_layer[l]):
					if l==hidden_layers: aux.append(list((t-s_output_layer[i])*s_output_layer[i]*(1-s_output_layer[i]))[i])
					else: 
						sum_aux = 0
						for r in xrange(neurons_by_layer[l+1]): sum_aux += delta[-1][r]*theta[l+1][r][i+1]
						aux.append(sum_aux*s_layers[l][i+1]*(1.0-s_layers[l][i+1]))			
				delta.append(np.array(aux))
				
			delta.reverse()
			
			for l in xrange(hidden_layers,-1,-1):
				for i in xrange(0,neurons_by_layer[l]):
					for j in xrange(len(theta[l][i])):
						if l!=0: theta[l][i][j] = theta[l][i][j]+ro*(delta[l][i]*s_layers[l-1][j])
						else:    theta[l][i][j] = theta[l][i][j] + ro*(delta[l][i]*x[j])
						
		if np.array_equal(theta_ant,theta): fixed=True
		
		theta_ant = []
		for i in xrange(hidden_layers+1):
			aux = []
			for j in xrange(neurons_by_layer[i]): aux.append(np.copy(theta[i][j]))
			theta_ant.append(aux)
			
		it += 1
		
	return theta

def incrementalBackwardPropagation(self): pass
def momentumBackwardPropagation(self): pass
def bufferBackwardPropagation(self): pass
	
if __name__ == "__main__":
	header()
	##### Definir topología de la red #####
	hidden_layers = 2 # N de capas ocultas #
	factivation   = [sigmoid,sigmoid,sigmoid] # Funcion de activacion #
	neurons_by_layer = [2,3,3] # Neuronas por capa #
	S = [(np.array([1,-2,1]),np.array([0,1,0])),(np.array([1,3,1]),np.array([1,0,0]))] # Conjunto de aprendizaje #
	# Theta iniciales (aleatorios si no se conoce bien el problema) #
	theta = [
				[np.array([1,1,1],dtype="f",copy=True),np.array([1,2,1],dtype="f",copy=True)],
				[np.array([1,1,1],dtype="f",copy=True),np.array([-1,-2,-1],dtype="f",copy=True),np.array([-1,2,-1],dtype="f",copy=True)],
				[np.array([1,1,1,1],dtype="f",copy=True),np.array([-1,-2,-1,1],dtype="f",copy=True),np.array([-1,2,-1,1],dtype="f",copy=True)]
			]
	max_iterations = 250 # Iteraciones maximas #
	ro             = 1   # Factor de aprendizaje)
	########################################
	
	
	##### Aprender parametros theta partiendo de un theta semilla #####
	theta = back_propagation(S,theta,hidden_layers,neurons_by_layer,factivation,ro,max_iterations)
	###################################################################
	
	
	##### Test muestra #####
	x             = np.array([1,3,1])
	print "Sample class: ",classify(x,theta,factivation)
	print "Regression result: ",regression(x,theta,factivation)
	########################
	
	footer()
