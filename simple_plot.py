#!/bin/python3

import csv
import matplotlib.pyplot as plt

names = [];
data = [];
with open('all_exp.txt', newline='') as csvfile:
  myreader = csv.reader(csvfile, delimiter=',')
  for idx,row in enumerate(myreader):
    if idx == 0:
      names = row
    else:
      data.append(row[2])

print("Found {0}".format(names))


fig = plt.figure()
plt.plot(data[::50])
plt.show()    
