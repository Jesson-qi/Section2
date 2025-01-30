#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def graham_scan(points):
	def orientation(p, q, r):
		return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

	points = sorted(points, key=lambda x: (x[0], x[1]))
	lower = []
	for p in points:
		while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)

	upper = []
	for p in reversed(points):
		while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)

	return np.array(lower[:-1] + upper[:-1])

def jarvis_march(points):
	n = len(points)
	if n < 3:
		return points

	hull = []
	l = np.argmin(points[:, 0])
	p = l
	while True:
		hull.append(points[p])
		q = (p + 1) % n
		for i in range(n):
			if (np.cross(points[i] - points[p], points[q] - points[p]) > 0):
				q = i
		p = q
		if p == l:
			break

	return np.array(hull)

def quickhull(points):
	def add_hull(points, p1, p2):
		if not len(points):
			return []

		distances = np.cross(points - p1, p2 - p1)
		farthest_point_index = np.argmax(distances)
		farthest_point = points[farthest_point_index]

		left_set_1 = points[np.cross(points - p1, farthest_point - p1) > 0]
		left_set_2 = points[np.cross(points - farthest_point, p2 - farthest_point) > 0]

		return add_hull(left_set_1, p1, farthest_point) + [farthest_point] + add_hull(left_set_2, farthest_point, p2)

	points = np.unique(points, axis=0)
	if len(points) < 3:
		return points

	leftmost = points[np.argmin(points[:, 0])]
	rightmost = points[np.argmax(points[:, 0])]

	above_set = points[np.cross(points - leftmost, rightmost - leftmost) > 0]
	below_set = points[np.cross(points - leftmost, rightmost - leftmost) < 0]

	upper_hull = add_hull(above_set, leftmost, rightmost)
	lower_hull = add_hull(below_set, rightmost, leftmost)

	return np.array([leftmost] + upper_hull + [rightmost] + lower_hull)

def monotone_chain(points):
	points = sorted(points, key=lambda x: (x[0], x[1]))

	lower = []
	for p in points:
		while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-1]) <= 0:
			lower.pop()
		lower.append(p)

	upper = []
	for p in reversed(points):
		while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-1]) <= 0:
			upper.pop()
		upper.append(p)

	return np.array(lower[:-1] + upper[:-1])

def showcase(points, function, title):
	plt.figure()
	plt.title(title)
	plt.scatter(points[:, 0], points[:, 1], color='blue')
	hull = function(points)
	hull = np.vstack((hull, hull[0]))
	plt.plot(hull[:, 0], hull[:, 1], color='red')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.show()
	plt.savefig(title)

if __name__ == '__main__':
	points = np.loadtxt('mesh.dat', skiprows=1)
	showcase(points, graham_scan, 'Graham scan')
	showcase(points, jarvis_march, 'Jarvis march')
	showcase(points, quickhull, 'Quickhull')
	showcase(points, monotone_chain, 'Monotone chain')