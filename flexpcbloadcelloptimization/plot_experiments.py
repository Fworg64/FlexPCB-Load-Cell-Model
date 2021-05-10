import sys
import argparse
import pdb
import csv

import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser(description="Specify the files and type(s) of plots")
  parser.add_argument("-c", "--convergence", help="Make a convergence plot", action='store_true')
  parser.add_argument("-f", "--force", help="Make a force plot", action="store_true")
  parser.add_argument('files', nargs='+')

  arg_dict = vars(parser.parse_args())
  series_names = []

  method_and_params = {}
  computation_time = {}
  x_star = {}
  obj_vals = {}
  force_vals = {}

  test_names = []

  for file in arg_dict["files"]:
    with open(file) as csvfile:
      csvreader = csv.reader(csvfile)
      temp_title = next(csvreader)
      title = "{0} {1}".format(temp_title[0], temp_title[1])
      test_names.append(title)
      method_and_params[title] = temp_title
      computation_time[title] = next(csvreader)
      x_star = next(csvreader)
      objective_history_title = next(csvreader)

      # iterate until you get to next title: "State History"
      obj_val = next(csvreader, None)
      pdb.set_trace()
      obj_vals[title] = []
      while "State History" not in obj_val:
        if obj_val is None:
          print("UH-Oh")
        obj_vals[title].append(obj_val[0])
        obj_val = next(csvreader, None)

      # Record force history
      force_val = next(csvreader, None)
      force_vals[title] = []
      while force_val is not None:
        force_vals[title].append(force_val[0])
        force_val = next(csvreader, None)
  pdb.set_trace()
  if arg_dict["convergence"]:
    plt.figure()
    for test in test_names:
      plt.plot(obj_vals[test])
      #plt.hold(True)
  
  plt.show()


if __name__ == "__main__":
  main()