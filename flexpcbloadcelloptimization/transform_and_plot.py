#!/bin/python3

import csv
import matplotlib.pyplot as plt
import sensordataproc.sensor_interp

names = [];
data = [];
time = [];


plot_index = 4;

with open('all_exp.txt', newline='') as csvfile:
  myreader = csv.reader(csvfile, delimiter=',')
  for idx,row in enumerate(myreader):
    if idx == 0:
      names = row
    else:
      data.append(row[plot_index])
      time.append(row[0])

print("Found {0}".format(names))

chan_params = {}
chan_params["L"] = 18e-6
chan_params["area"] = 0.001164
chan_params["cfilt"] = 33e-12
chan_params["er"] = 3.3

chan_dist = calculate_distance_from_readings_and_params(data, chan_params)


fig = plt.figure()
plt.plot(time[:1000:50], chan_dist[:1000:50])
plt.title(names[plot_index])
plt.show()    
