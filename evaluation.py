import os
import numpy as np


def list_csv_files_recursive(folder_path):
    csv_files = []
    for foldername, subfolder, filenames in os.walk(folder_path):
        temp_csv = []
        for filename in filenames:
            if filename.endswith(".csv"):
                temp_csv.append(os.path.join(foldername, filename))
        if len(temp_csv) == 2:
            temp_csv.sort()
            csv_files.append(temp_csv)
    csv_files.sort()
    return csv_files


def readCSV(path):
    """
    This function reads the CSV file and returns a list of lists
    """
    with open(path, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split(",") for x in data]
    # remove the header
    data = data[1:]
    # convert the data to double
    data = [[float(y) for y in x] for x in data]
    # get rid of the first column
    data = [x[1:] for x in data]
    # turn to numpy array
    data = np.array(data)
    return data


# calculate MAE
def calcMAE(trans, truth):
    trans = readCSV(trans)
    truth = readCSV(truth)
    absolute_errors = np.abs(trans - truth)
    mae = np.mean(absolute_errors)

    return mae


# calculate Median Square Error
def calcMedianAE(trans, truth):
    trans = readCSV(trans)
    truth = readCSV(truth)
    absolute = np.abs(trans - truth)
    print(trans)
    print(truth)
    print(absolute)
    return np.median(absolute)


if __name__ == "__main__":
    # folder with all your data & csv
    interestedFolder = "./BraTSReg_Training_Data_v3/"
    results = interestedFolder + "csv/"

    # recursively find all csv files
    csv_files = list_csv_files_recursive(interestedFolder)
    target_csv = [csv[1] for csv in csv_files]
    index = [i - 1 for i in [2, 4, 5, 10, 12, 16, 22, 24, 29, 67]]
    target_csv = [target_csv[i] for i in index]
    results_csv = []
    for root, dirs, files in os.walk(results):
        # 'root' is the current directory being processed
        # 'dirs' is a list of subdirectories in 'root'
        # 'files' is a list of files in 'root'
        for filename in files:
            # Process each file here
            full_path = os.path.join(root, filename)
            results_csv.append(full_path)
    results_csv.sort()
    print(results_csv)

    # calculate MAE for each pair of csv files

    result = [calcMedianAE(f, f_h) for f, f_h in zip(target_csv, results_csv)]
    sum = 0
    for r in result:
        sum += r
    print(sum / 10)
    print(result)

    # print the result
    # print(result)
