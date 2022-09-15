#import relevant libaries
import numpy as np
import sys
from casadi import *

import do_mpc

#define model type and intialize
model_type = 'discrete'
model = do_mpc.model.Model(model_type = model_type, symvar_type = "MX")

#define (certain) parameters of the system
A = np.array([
    [1, -6.66e-13, -2.03e-9, -4.14e-6],
    [9.83e-4, 1, -4.09e-8, -8.32e-5],
    [4.83e-7, 9.83e-4, 1, -5.34e-4],
    [1.58e-10, 4.83e-7, 9.83e-4, .9994]
])

B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T

C = np.array([[-.0096, .0135, .005, -.0095]])

D = np.zeros((1, 1))

#define desired values for x and y- copied from Jake's code
def y_to_z(y):
    return np.exp(61.4*y - 5.468)

def z_to_y(z):
    return (np.log(z) + 5.468)/61.4

zD = 0.2
inv = np.linalg.inv
I = np.eye(4)
yD = z_to_y([zD])
uD = inv(C@inv(I - A)@B) @ yD
xD = inv(I - A) @ B @ uD

#define variables for the state
_x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

#define input
_u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

#Define LDS
x_next = A@_x + B@_u  

#set up y and z for measurement and cost function
_y = C@_x    

#Add LDS to model using "right-hand-side" method
model.set_rhs('x', x_next)

_Z = y_to_z(_y)

#set cost function to be used by controller later
model.set_expression(expr_name='cost', expr=(((1/100)*(yD - _y))**2)) 

#add z as an "auxilliary" expression to plot later
model.set_expression(expr_name='Z', expr=_Z)

#this finalizes the model
model.setup()

#initialize the controller using the MPC class which uses the above model object
mpc = do_mpc.controller.MPC(model)

#setting values for the prediction horizon, time step, etc
#time step for use in cntrl and also simulation later
time_step = 0.001

setup_mpc = {
    'n_robust': 0, #this would be > 0 if we were using a branch scheme for uncertain parameters
    'n_horizon': 7, 
    't_step': time_step, #trying this value as neuron firing can be on the order of 10e-3 seconds
    'state_discretization': 'discrete',
    'store_full_solution':False,    
    'store_lagr_multiplier': False,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}             #last two comments are from their documentation to revisit later
}

mpc.set_param(**setup_mpc)

#Give the above cost funtion to the controller 
mterm = model.aux['cost'] # terminal cost       
lterm = model.aux['cost'] # terminal cost
 # stage cost                                   #what do they mean by terminal and stage cost?

mpc.set_objective(mterm=mterm, lterm=lterm)     

mpc.set_rterm(u=10**(-4))       #see 'getting started: MPC" for l, m, and r terms

#Now we give the controller contraints
#state bounds
    #None for now

#input bounds
mpc.bounds['lower','_u','u'] = 0 
mpc.bounds['upper','_u','u'] = 4.228704824986444    #these values were taken from lqr.py

#finalize our controller
mpc.setup()

#Estimator configured assuming all states can be directly measured 
estimator = do_mpc.estimator.StateFeedback(model) 

#Creating a simultor to run MPC in a closed loop
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = time_step) #same time step as above for controller
simulator.setup()

#Now to run the closed loop control
#setting an initial state identical to one from lqr.py
x0 = np.array([[1.43686192], [-1.45507814], [-1.70993785], [-1.59286133]])     

#give initial value to components of the control loop
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess
mpc.set_initial_guess()

#then run loop
#additional function to supress lengthy output from controller
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
with  suppress_stdout():
    #here is the actual closed loop
    for i in range(10000):
        u0 = mpc.make_step(x0)
        y1 = simulator.make_step(u0)
        x0 = estimator.make_step(y1)
    
#plotting the results
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

import matplotlib.pyplot as plt
fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))

graphics.plot_results()
graphics.reset_axes()
plt.show()

# #for saving results and looking at individual values
# from do_mpc.data import save_results

# save_results([mpc, simulator], overwrite=True) #- commented out so we don't make duplicates
