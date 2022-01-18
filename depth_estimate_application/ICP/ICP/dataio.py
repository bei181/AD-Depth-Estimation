import numpy as np

def loadData(fileName):
    fin = open(fileName, "r")
    lines = fin.readlines()
    cnt = len(lines)
    data = np.zeros([cnt,3])
    for i in range(2,cnt):
        data[i - 2] = lines[i].split(' ')
    fin.close()
    return data

def outputData(fileName, data):
    fout = open(fileName, "w")
    for i in range(data.shape[0]):
        fout.write(str(data[i][0]) + ',' + str(data[i][1]) + ',' + str(data[i][2]) + '\n')
    fout.close()
