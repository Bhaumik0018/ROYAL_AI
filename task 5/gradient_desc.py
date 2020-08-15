# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:06:19 2020

@author: bhaumik
"""

import time
import numpy as np
epoch=1000

X =[0.5,2.5]
Y =[0.2,0.9]

def f(w,b,x):
  return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):
  err= 0.0
  for x,y in zip(X,Y):
    fx = f(w,b,x)
    err += 0.5 * (fx-y)**2
  return err


def grad_b(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)

def grad_w(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)*x

def do_nesterov_gradient():
    w,b,eta=-2,-2,5
    prev_v_w,prev_v_b,gamma=0,0,0.1
    for i in range(epoch):
        dw,db=0,0
        v_w=gamma*prev_v_w
        v_b=gamma* prev_v_b
        for x,y in zip(X, Y):
            dw += grad_w(w-v_w,b-v_b,x, y)
            db += grad_b(w-v_w,b-v_b,x, y)
        
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        w=w-v_w
        b=b- v_b
        prev_v_w=v_w
        prev_v_b=v_b
        
    print("nestrov dg error=",error(w, b))    
    print("nestrov final weight=",w)
    print("nestrov final bias=",b)
    print("nestrov final epoch=",epoch)
            

def do_momentum_gradient():
    w,b,eta=-2,-2,5
    prev_v_w,prev_v_b,gamma=0,0,0.1
    for i in range(epoch): 
        dw,db=0,0
        for x,y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
     
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        w=w-v_w
        b=b-v_b
        prev_v_w=v_w
        prev_v_b= v_b
    
    print("momentum dg error=",error(w, b))    
    print("momentum final weight=",w)
    print("momentum final bias=",b)
    print("momentum final epoch=",epoch)
        
def do_gradient_descent():
  w,b,eta= -2,-2,5             
  for i in range(epoch):
    dw=0
    db=0
    for x,y in zip(X,Y):
      dw += grad_w(w,b,x,y)
      db += grad_b(w,b,x,y)
    w = w - eta *dw
    b = b - eta * db
  print("gd=",error(w,b)) 
  print("gd final weight=",w)
  print("gd final bias=",b)
  print("gd final epoch=",epoch)

times=list()

start_time = time.process_time()
do_gradient_descent()
stop_time1 = time.process_time()

times.append(stop_time1-start_time)
print("time=",times)




stop_time2 = time.process_time()
do_momentum_gradient()
stop_time3 = time.process_time()


times.append(stop_time3-stop_time2)
print("time=",times)


stop_time4 = time.process_time()
do_nesterov_gradient()
stop_time5 = time.process_time()

times.append(stop_time4-stop_time5)
print("time=",times)

