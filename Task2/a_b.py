#!/usr/bin/env python

from collections import Counter
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def triangle_area(x, y):
	return np.cross(x[1:] - x[0], y[1:] - y[0])/2

def surface1(x, y):
	return 2*x**2 + 2*y**2

def surface2(x, y):
	return 2*np.exp(-x**2 - y**2)

x = np.linspace(-1, 1, 21)
y = np.linspace(-1, 1, 21)
x, y = np.meshgrid(x, y)
indices = surface1(x, y) <= surface2(x, y)
n = indices.sum()
x = x[indices]
y = y[indices]
triangles = Delaunay(np.c_[x, y]).simplices
triangles = [v for v in triangles if not np.isclose(triangle_area(x[v], y[v]), 0)]

edge_counter = Counter()
for vertices in triangles:
	i1, i2, i3 = sorted(vertices)
	edge_counter.update([(i1, i2), (i2, i3), (i1, i3)])
boundary = np.unique([edge for edge, count in edge_counter.items() if count == 1])
inner = np.setdiff1d(np.arange(n), boundary)

all_x = np.concatenate([x, x])
all_y = np.concatenate([y, y])
all_z = np.concatenate([surface1(x, y), surface2(x, y)])
all_triangles = np.concatenate([triangles, [[i if i in boundary else i+n for i in v] for v in triangles]])

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(all_x, all_y, all_z, triangles=all_triangles, cmap=cm.viridis)
plt.show()
plt.savefig("c.png")
#!/usr/bin/env python

from collections import Counter
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def surface1(x, y):
	return 2*x**2 + 2*y**2

def surface2(x, y):
	return 2*np.exp(-x**2 - y**2)

x = np.linspace(-1, 1, 51)
y = np.linspace(-1, 1, 51)
z = np.linspace(0, 2, 21)
x, y, z = np.meshgrid(x, y, z)
indices = np.logical_and(surface1(x, y) <= z, z <= surface2(x, y))
x = x[indices]
y = y[indices]
z = z[indices]
tetrahedra = Delaunay(np.c_[x, y, z]).simplices

face_counter = Counter()
for vertices in tetrahedra:
	i1, i2, i3, i4 = sorted(vertices)
	face_counter.update([(i1, i2, i3), (i1, i2, i4), (i1, i3, i4), (i2, i3, i4)])
boundary_faces = [face for face, count in face_counter.items() if count == 1]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z, triangles=boundary_faces, cmap=cm.viridis)
plt.show()
plt.savefig("d.png")