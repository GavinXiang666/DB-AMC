import numpy as np
import math


def update_position(vector,old_origin_point,new_origin_point):
    new_origin_x = new_origin_point[0]
    new_origin_y = new_origin_point[1]
    new_origin_z = new_origin_point[2]
    changed_vectors = []
    for change_vector in vector:
        delta_x = new_origin_x - old_origin_point[0]
        delta_y = new_origin_y - old_origin_point[1]
        delta_z = new_origin_z - old_origin_point[2]
        change_vector = np.array([change_vector[0]-delta_x,change_vector[1]-delta_y,change_vector[2]-delta_z])
        
        changed_vectors.append(change_vector)
    
    return changed_vectors
v = [np.array([4,3,0]),np.array([1,1,2]),np.array([1,5,0])]
new_o = np.array([4,3,0])
result = update_position(v,old_origin_point=np.array([0,0,0]),new_origin_point=new_o)
print(result)