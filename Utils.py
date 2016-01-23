#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_factivation_layer(factivation,derivative_factivation,layer_units):
	""" Genera el conjunto de funciones de activacion y sus derivadas para 1 capa completa """
	return [(factivation,derivative_factivation)]*layer_units

def get_connections(units_by_layer): return sum([units_by_layer[i]*units_by_layer[i+1] for i in xrange(len(units_by_layer)-1)])+units_by_layer[len(units_by_layer)-1]

def get_layers(units_by_layer): return len(units_by_layer)

def get_info(units_by_layer,factivation):
	print """\n			  ************************
			  Estructura de la red neuronal
			  ************************ \n\n"""
	print "Capas ocultas (incluida capa de salida): ",get_layers(units_by_layer)
	print "Numero de conexiones: ",get_connections(units_by_layer)
	for l in xrange(get_layers(units_by_layer)):
		print "\n----------\nCapa: ",l,"\n----------\n"
		print "\tNeuronas en la capa -> ",units_by_layer[l],"\n"
		if l!=0: print "\tFunciones de activacion -> ",factivation[l-1],"\n"
	print "\n\n-------------------------------------------------------\n\n"
