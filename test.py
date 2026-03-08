from wire import wire_shape
import math
points = []
for i in  range(10):
    a = [1,1 + math.cosh(i/10)]
    points.append(a)
wire_shape(points,None ,1, 1)