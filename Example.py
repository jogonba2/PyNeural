#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import MLPLearning,ActivationFuncts,Utils,Config,Decision
import logging 
import numpy as np

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	Config.header()
	# Muestras de entrenamiento #
	X = [np.array([1.0,-0.5,1.0],dtype="f",copy=True),
		 np.array([1.0,-2.0,1.0],dtype="f",copy=True),
		 np.array([1.0,-5.0,1.0],dtype="f",copy=True),
		 np.array([1.0,-7.0,1.0],dtype="f",copy=True),
		 np.array([1.0,10.0,1.0],dtype="f",copy=True),
		 np.array([1.0,8.0,5.0],dtype="f",copy=True)]
	Y = [np.array([1,0,0],dtype="f",copy=True),
		 np.array([1,0,0],dtype="f",copy=True),
		 np.array([0,1,0],dtype="f",copy=True),
		 np.array([0,1,0],dtype="f",copy=True),
		 np.array([0,0,1],dtype="f",copy=True),
		 np.array([0,0,1],dtype="f",copy=True)]
	S = [(X[i],Y[i]) for i in xrange(len(X))]
	############################
	
	# Topologia de la red #
	units_by_layer = [len(S[0][0]),2,3]
	factivation    = [Utils.get_factivation_layer(ActivationFuncts.sigmoid,ActivationFuncts.derivative_sigmoid,units_by_layer[1]),
					  Utils.get_factivation_layer(ActivationFuncts.lineal,ActivationFuncts.derivative_lineal,units_by_layer[2])]
	Utils.get_info(units_by_layer,factivation)
	#######################
	
	# Plot #
	
	"""pca = PCA(n_components=2)
	pca.fit(X)
	X = pca.transform(X)
	S = [(np.array([1]+X[i].tolist()),Y[i]) for i in xrange(len(X))]
	print S"""
	
	########
	
	########
	# Aprender vector theta      #
	rho = 0.5
	nu  = 0.5
	l   = 1
	theta1 = MLPLearning.back_propagation_batch(S,rho,units_by_layer,factivation,850,50)
	theta2 = MLPLearning.back_propagation_online(S,rho,units_by_layer,factivation,850,50)
	theta3 = MLPLearning.back_propagation_batch_momentum(S,rho,nu,units_by_layer,factivation,850,50)
	theta4 = MLPLearning.back_propagation_batch_buffer(S,rho,l,units_by_layer,factivation,850,50)
	theta5 = MLPLearning.back_propagation_online_buffer(S,rho,l,units_by_layer,factivation,850,50)
	theta6 = MLPLearning.back_propagation_online_momentum(S,rho,nu,units_by_layer,factivation,850,50)
	theta7,fitness = MLPLearning.evolutional(S,units_by_layer,factivation,200,500,-2,2,1.1,0.9)
	##############################
	
	# Clasificacion #
	logging.info("Clase con theta1: (Backprop batch): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta1,factivation)))
	logging.info("Clase con theta2: (Backprop online): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta2,factivation)))
	logging.info("Clase con theta3: (Backprop batch con momentum): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta3,factivation)))
	logging.info("Clase con theta4: (Backprop batch con amortiguamiento): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta4,factivation)))
	logging.info("Clase con theta5: (Backprop online con amortiguamiento): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta5,factivation)))
	logging.info("Clase con theta6: (Backprop online con momentum): "+str(Decision.classify(units_by_layer,[1.0,-6.3,1.0],theta6,factivation)))
	logging.info("Clase con theta7: (Algoritmo genetico): "+str(Decision.classify(units_by_layer,[1.0,-5.0,1.0],theta7,factivation))+" fitness: "+str(fitness))

	#################
	# Regresion #
	logging.info("Regresion con theta1: (Backprop batch): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta1,factivation)))
	logging.info("Regresion con theta2: (Backprop online): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta2,factivation)))
	logging.info("Regresion con theta3: (Backprop batch con momentum): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta3,factivation)))
	logging.info("Regresion con theta4: (Backprop batch con amortiguamiento): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta4,factivation)))
	logging.info("Regresion con theta5: (Backprop online con amortiguamiento): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta5,factivation)))
	logging.info("Regresion con theta6: (Backprop online con momentum): "+str(Decision.regression(units_by_layer,[1.0,-6.3,1.0],theta6,factivation)))
	logging.info("Regresion con theta7: (Algoritmo genetico): "+str(Decision.regression(units_by_layer,[1.0,34.5,37.3],theta7,factivation))+" fitness: "+str(fitness))
	#############
