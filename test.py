import numpy as np

"""
a = np.array([[1,2],[3,4]],dtype=np.float32)
b = np.array([[4,3],[2,1]],dtype=np.float32)
c = b>a
print(c.dtype)
d = np.sum(c,axis=0)
print(d)
print(d.dtype)
"""
"""
tu = (1,2)
tuy,tux = tu
print(tux)
print(tuy)
"""
"""
a = np.array([5,4,3,2,1,1,1,5,5,7])
print(np.argmin(a))
"""
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a[:,1:-1])