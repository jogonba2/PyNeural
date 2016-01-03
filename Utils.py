#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_factivation_layer(factivation,derivative_factivation,layer_units):
	""" Genera el conjunto de funciones de activacion y sus derivadas para 1 capa completa """
	return [(factivation,derivative_factivation)]*layer_units

def get_info(units_by_layer,factivation):
	print """\n			  ************************
			  Estructura de la red neuronal
			  ************************ \n\n"""
	connections = sum([units_by_layer[i]*units_by_layer[i+1] for i in xrange(len(units_by_layer)-1)])+units_by_layer[len(units_by_layer)-1]
	print "Capas ocultas (incluida capa de salida): ",len(units_by_layer)
	print "Numero de conexiones: ",connections
	for l in xrange(len(units_by_layer)):
		print "\n----------\nCapa: ",l,"\n----------\n"
		print "\tNeuronas en la capa -> ",units_by_layer[l],"\n"
		if l!=0: print "\tFunciones de activacion -> ",factivation[l-1],"\n"
	print "\n\n-------------------------------------------------------\n\n"
