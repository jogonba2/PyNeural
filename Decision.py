#!/usr/bin/env python
# -*- coding: utf-8 -*-
import MLPLearning

def classify(units_by_layer,xk,theta,factivation):
	phi,s = MLPLearning.forward_propagation(units_by_layer,xk,theta,factivation)
	return s[-1].index(max(s[-1]))

def get_output_vector(units_by_layer,xk,theta,factivation):
	phi,s = MLPLearning.forward_propagation(units_by_layer,xk,theta,factivation)
	return s[len(units_by_layer)-1]
	
def regression(units_by_layer,xk,theta,factivation):
	phi,s = MLPLearning.forward_propagation(units_by_layer,xk,theta,factivation)
	return s[-1]
