#!/bin/python3

import csv
import matplotlib.pyplot as plt

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


fig = plt.figure()
plt.plot(time[:1000:50], data[:1000:50])
plt.title(names[plot_index])
plt.show()    
