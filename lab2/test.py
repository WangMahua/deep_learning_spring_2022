import numpy as np 
a = np.array([[1,2],[3,4]])
b = np.array([[4,2],[3,1]])



y = np.array([1,2,3,4,5,6])
# pred_y = np.array([4,2])
# def loss_function(y,pred_y):
#     return ((y - pred_y)**2).mean(axis=0)
# r = loss_function(y,pred_y)
# print(type(r))
# print(type(np.sum(r)))
# mse = (np.linalg.norm(y-pred_y)**2)/len(y)
# print(type(float(mse)))
# print(loss_function(y,pred_y))
print(y.reshape(2,3))