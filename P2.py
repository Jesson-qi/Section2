#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import timeit

from P1 import graham_scan, jarvis_march, quickhull, monotone_chain

def point_cloud(n, min=0, max=1):
	return np.random.rand(n, 2) * (max - min) + min

def time_algorithm(algorithm, points):
	return timeit.timeit(lambda: algorithm(points), number=1)

def run_time(n_list, min=0, max=1):
	graham_scan_times = []
	jarvis_march_times = []
	quickhull_times = []
	monotone_chain_times = []
	for n in n_list:
		points = point_cloud(n, min, max)
		graham_scan_times.append(time_algorithm(graham_scan, points))
		jarvis_march_times.append(time_algorithm(jarvis_march, points))
		quickhull_times.append(time_algorithm(quickhull, points))
		monotone_chain_times.append(time_algorithm(monotone_chain, points))
	return graham_scan_times, jarvis_march_times, quickhull_times, monotone_chain_times

def plot_time_complexity(n_list, min=0, max=1):
	graham_scan_times, jarvis_march_times, quickhull_times, monotone_chain_times = run_time(n_list, min, max)
	plt.figure()
	plt.plot(n_list, graham_scan_times, label="Graham scan")
	plt.plot(n_list, jarvis_march_times, label="Jarvis march")
	plt.plot(n_list, quickhull_times, label="Quickhull")
	plt.plot(n_list, monotone_chain_times, label="Monotone chain")
	plt.legend()
	plt.xlabel("Number of Points")
	plt.ylabel("Time (s)")
	plt.title(f'Point cloud bounded in $[{min},{max}]^2$')
	plt.show()
	plt.savefig(f"time_complexity_{min}_{max}.png")

def plot_time_histogram(n, count, min=0, max=1):
	times_list = run_time(np.full(count, n), min, max)
	for times, title in zip(times_list, ["Graham scan", "Jarvis march", "Quickhull", "Monotone chain"]):
		plt.figure()
		plt.hist(times, bins=20)
		plt.title(title)
		plt.xlabel("Time (s)")
		plt.ylabel("Frequency")
		plt.show()
		plt.savefig(f"P2_histogram_{title}.png")

if __name__ == "__main__":
	np.random.seed(1108)

	n_list = [10,50,100,200,400,800,1000]
	plot_time_complexity(n_list)
	plot_time_complexity(n_list, -5, 5)
	plot_time_histogram(50, 100)