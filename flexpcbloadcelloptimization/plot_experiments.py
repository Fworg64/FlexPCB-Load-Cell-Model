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
  GT_vals = {}
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
      obj_vals[title] = []
      while "State History" not in obj_val:
        if obj_val is None:
          print("UH-Oh")
        obj_vals[title].append(float(obj_val[0]))
        obj_val = next(csvreader, None)

      # Record force history
      force_val = next(csvreader, None)
      force_vals[title] = []
      while "Ground Truth" not in force_val:
        force_vals[title].append(float(force_val[0]))
        force_val = next(csvreader, None)

      # Record Ground Truth
      GT_val = next(csvreader, None)
      GT_vals[title] = []
      while GT_val is not None:
        GT_vals[title].append(float(GT_val[0]))
        GT_val = next(csvreader, None)
  if arg_dict["convergence"]:
    fig1 = plt.figure(1)
    for test in test_names:
      plt.plot(obj_vals[test], label=test)

    fig1.legend()
  if arg_dict["force"]:
    fig2 = plt.figure(2)
    for test in test_names:
      plt.plot(force_vals[test], label=test)
    plt.plot(GT_vals[test_names[0]], label='Ground Truth')
    fig2.legend()
  plt.show()


if __name__ == "__main__":
  main()