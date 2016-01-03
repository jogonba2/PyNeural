#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import warnings
import Config
import Decision
warnings.filterwarnings("ignore")

def forward_propagation(units_by_layer,xk,theta,factivation):
	phi = []
	s   = []
	# Inicializar primera capa #
	phi.append([])
	s.append([])
	for i in xrange(units_by_layer[1]):
		aux_phi = 0
		for j in xrange(len(xk)): aux_phi += theta[0][i][j]*xk[j]
		phi[-1].append(aux_phi)
		s[-1].append(factivation[0][i][0](aux_phi))
	# Resto de capas #
	for l in xrange(2,len(units_by_layer)):
		phi.append([])
		s.append([])
		for i in xrange(units_by_layer[l]):
			aux_phi = theta[l-1][i][0]
			for j in xrange(units_by_layer[l-1]): 
				aux_phi += theta[l-1][i][j+1]*s[-2][j]			
			phi[-1].append(aux_phi)
			s[-1].append(factivation[l-1][i][0](aux_phi))
	s.insert(0,xk)
	
	return phi,s
		
def back_propagation_batch(S,rho,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))		
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	# Mientras no converja #
	while k<max_it:
		# Inicializar incr_theta #
		incr_theta = []
		for l in xrange(1,len(units_by_layer)):
			incr_theta.append([])
			if l-1==0: sm = units_by_layer[l-1]
			else:	   sm = units_by_layer[l-1]+1
			for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		for (xk,tk) in S:
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################	
		# Actualizar pesos #					
		for l in xrange(len(theta)):
			for i in xrange(len(theta[l])): theta[l][i] += rho*incr_theta[l][i]
		
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop batch: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk],label=predicted_tk)
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta


def back_propagation_online(S,rho,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	# Mientras no converja #
	while k<max_it:
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		m = 0	   # N muestra
		for (xk,tk) in S: 
			# Inicializar incr_theta a cada muestra #
			incr_theta = []
			for l in xrange(1,len(units_by_layer)):
				incr_theta.append([])
				if l-1==0: sm = units_by_layer[l-1]
				else:	   sm = units_by_layer[l-1]+1
				for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
			##########################
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################
			# Actualizar pesos #					
			for l in xrange(len(theta)):
				for i in xrange(len(theta[l])): theta[l][i] += rho*incr_theta[l][i]
			m += 1	
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop online: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk])
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta

def back_propagation_batch_momentum(S,rho,nu,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))
		
	# Inicializar incr_theta_ant #
	incr_theta_ant = []
	for l in xrange(1,len(units_by_layer)):
		incr_theta_ant.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): incr_theta_ant [-1].append(np.zeros(sm))
	
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	
	# Mientras no converja #
	while k<max_it:
		# Inicializar incr_theta #
		incr_theta = []
		for l in xrange(1,len(units_by_layer)):
			incr_theta.append([])
			if l-1==0: sm = units_by_layer[l-1]
			else:	   sm = units_by_layer[l-1]+1
			for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		for (xk,tk) in S: 
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################					
		# Actualizaciones #					
		for l in xrange(len(theta)):
			for i in xrange(len(theta[l])): 	
				# Actualizar los pesos #	
				theta[l][i] += (rho*incr_theta[l][i]) + (nu*incr_theta_ant[l][i])
				# Actualizar incr_theta_ant #
				incr_theta_ant[l][i] = (rho*incr_theta[l][i]) + (nu*incr_theta_ant[l][i])
		
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop batch con momentum: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk])
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta
	
def back_propagation_batch_buffer(S,rho,l,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))
		
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	
	# Mientras no converja #
	while k<max_it:
		# Inicializar incr_theta #
		incr_theta = []
		for l in xrange(1,len(units_by_layer)):
			incr_theta.append([])
			if l-1==0: sm = units_by_layer[l-1]
			else:	   sm = units_by_layer[l-1]+1
			for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		for (xk,tk) in S: 
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################
						
		# Actualizar pesos #					
		for l in xrange(len(theta)):
			for i in xrange(len(theta[l])): theta[l][i] += (rho*incr_theta[l][i]) + (2*rho*l*theta[l][i])
		
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop batch con amortiguamiento: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk])
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta

def back_propagation_online_buffer(S,rho,l,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))
	
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	
	# Mientras no converja #
	while k<max_it:
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		m = 0	   # N muestra
		for (xk,tk) in S: 
			# Inicializar incr_theta a cada muestra #
			incr_theta = []
			for l in xrange(1,len(units_by_layer)):
				incr_theta.append([])
				if l-1==0: sm = units_by_layer[l-1]
				else:	   sm = units_by_layer[l-1]+1
				for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
			##########################
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################
			# Actualizar pesos #					
			for l in xrange(len(theta)):
				for i in xrange(len(theta[l])): theta[l][i] += (rho*incr_theta[l][i]) + (2*rho*l*theta[l][i])
			m += 1	
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop online con amortiguamiento: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk])
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta

def back_propagation_online_momentum(S,rho,nu,units_by_layer,factivation,max_it=250,report_it=50):
	k = 0      # it
	delta = [] # Errores  
	
	# Inicializar pesos a 0 #
	theta = []
	for l in xrange(1,len(units_by_layer)):
		theta.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): theta[-1].append(np.zeros(sm))
				
	# Inicializar incr_theta_ant #
	incr_theta_ant = []
	for l in xrange(1,len(units_by_layer)):
		incr_theta_ant.append([])
		if l-1==0: sm = units_by_layer[l-1]
		else:	   sm = units_by_layer[l-1]+1
		for i in xrange(units_by_layer[l]): incr_theta_ant [-1].append(np.zeros(sm))
		
	# Plot #
	if Config.PLOT: color_classes = [Config.COLORS[c%len(Config.COLORS)]+Config.STYLE[c%len(Config.STYLE)] for c in xrange(len(S[0][1]))]
	
	# Mientras no converja #
	while k<max_it:
		# Inicializar delta #
		delta = []
		for l in xrange(1,len(units_by_layer)):
			delta.append([])
			for i in xrange(units_by_layer[l]): delta[-1].append(0)
		# Para cada muestra #
		m = 0	   # N muestra
		for (xk,tk) in S: 
			# Inicializar incr_theta a cada muestra #
			incr_theta = []
			for l in xrange(1,len(units_by_layer)):
				incr_theta.append([])
				if l-1==0: sm = units_by_layer[l-1]
				else:	   sm = units_by_layer[l-1]+1
				for i in xrange(units_by_layer[l]): incr_theta[-1].append(np.zeros(sm))
			##########################
			phi,s = forward_propagation(units_by_layer,xk,theta,factivation)
			# Desde la salida a la entrada #
			for l in xrange(len(units_by_layer)-1,0,-1):
				# Para cada nodo #
				for i in xrange(units_by_layer[l]):
					########## Calcular delta ###########	
					if l==len(units_by_layer)-1: delta[l-1][i] = (factivation[l-1][i][1](phi[l-1][i])*(tk[i]-s[l][i]))
					else: delta[l-1][i] += factivation[l-1][i][1](phi[l-1][i])*sum([delta[l][r]*theta[l][r][i+1] for r in xrange(units_by_layer[l+1])])					
					#####################################
					if l==len(units_by_layer)-1:
						# Calcular incr_theta (capa salida) #
						incr_theta[l-1][i][0] += delta[l-1][i]
						for j in xrange(units_by_layer[l-1]): incr_theta[l-1][i][j+1] += delta[l-1][i]*s[l-1][j]
						#####################################	
					else:
						# Calcular incr_theta (capas ocultas) #
						for j in xrange(units_by_layer[l-1]):
							if j==0: incr_theta[l-1][i][j] += delta[l-1][i]
							else:    incr_theta[l-1][i][j] += delta[l-1][i]*s[l-1][j]
						#####################################
			# Actualizaciones #					
			for l in xrange(len(theta)):
				for i in xrange(len(theta[l])): 	
					# Actualizar los pesos #	
					theta[l][i] += (rho*incr_theta[l][i]) + (nu*incr_theta_ant[l][i])
					# Actualizar incr_theta_ant #
					incr_theta_ant[l][i] = (rho*incr_theta[l][i]) + (nu*incr_theta_ant[l][i])
			m += 1	
		
		# Plot de las muestras #
		if Config.PLOT:
			if k%report_it==0:
				plt.clf()
				plt.ylabel("Y")
				plt.xlabel("X")
				maxX = float("-inf")
				maxY = float("-inf")
				plt.title("Iteracion backprop online con momentum: "+str(k))
				for (xk,tk) in S:
					predicted_tk = Decision.classify(units_by_layer,xk,theta,factivation)
					plt.plot(xk[1],xk[2],color_classes[predicted_tk])
					if xk[1]>maxX: maxX = xk[1]
					if xk[2]>maxY: maxY = xk[2]
				plt.axis([-maxX,2*maxX,-maxY,2*maxY])
				plt.show(block=False)
				print "Siguientes ",report_it," iteraciones [Enter]"
				raw_input()
				plt.close()
		k += 1
	return theta

def evolutionary(S,rho,nu,units_by_layer,factivation,max_it=250,report_it=50): pass
