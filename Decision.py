#!/usr/bin/env python
# -*- coding: utf-8 -*-
import MLPLearning

def classify(units_by_layer,theta,xk,factivation):
	phi,s = MLPLearning.forward_propagation(units_by_layer,theta,xk,factivation)
	return s[-1].index(max(s[-1]))

def regression(units_by_layer,theta,xk,factivation):
	phi,s = MLPLearning.forward_propagation(units_by_layer,theta,xk,factivation)
	return s[-1]
