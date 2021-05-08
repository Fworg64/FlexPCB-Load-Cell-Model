import numpy as np
import csv

def read_experiment_data(path):
  """ 
  Given the path to the all_exp.txt file, loads the time series data and returns a dictionary of the columns.
  """
  with open('all_exp.txt', newline='') as csvfile:
    myreader = csv.reader(csvfile, delimiter=',')
    for idx,row in enumerate(myreader):
      if idx == 0:
        names = row
        data = {name: [] for name in names}
      else:
        for coldex, name in enumerate(names):
          data[name].append(row[coldex])
  return data
   

def calculate_distance_from_readings_and_params(chan_data_in, params):
  """
  chan_data : The list/array of read values for a particular channel
  
  params    : A dictionary with the following entries
       "area" : The area in m^2 common to the plates
        "L"   : The inductance of the LC circuit
      "cfilt" : The additional capcitance not attributed to the sensor
        "er"  : The relative permitivity of the dielectric
  """

  e0 = 8.854e-12 # Dielectric permitivity of free space
  pi = 3.1415

  chan_data = np.asarray([float(datum) for datum in chan_data_in])

  # Resonant frequency of tank ciruit is [clk_freq (40 MHz)] x reading / [gain (2^16)]
  sensor_reading_freq = 40e6 * chan_data / (2**16); 
  # Capacitance is given from resonant frequency as
  sensor_reading_cap  = 1.0 / (params["L"] * (2.0*pi*sensor_reading_freq)**2.0) - params["cfilt"];
  # Distance is given from parallel plate formula as
  sensor_reading_dist = e0*params["er"]*params["area"] / sensor_reading_cap;

  return sensor_reading_dist



