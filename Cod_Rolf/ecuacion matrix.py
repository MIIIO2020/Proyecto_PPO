import numpy as np 
a = np.array([[0.37,-0.27,0,0],
              [0,0.37,-0.27,0],
              [-0.63,-0.27,0,1]]) 
print('Array a:') 
print(a) 
ainv = np.linalg.inv(a) 

print('Inverse of a:')
print( ainv )

print ('Matrix B is:')
b = np.array([100,80,50,-100]) 
print (b )

print ('Compute A-1B:' )
x = np.linalg.solve(a,b) 
print (x  )

# this is the solution to linear
# equations x = 5, y = 3, z = -2