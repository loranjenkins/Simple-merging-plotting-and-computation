import pandas as pd
import numpy as np
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from trackobjects.trackside import TrackSide

if __name__ == '__main__':
    data = pd.read_csv(r'D:\Pycharmprojects\Simple-merging-plotting-and-computation\data_folder\joan_data_experment_2vehicles.csv', sep = ';')

    transform = data.iloc[:,2]
    print(transform)
    print(transform[0][2])


    fig = plt.figure()

    # plt.plot(data['travelled_distance'][side], data['velocities'][side], label=str(side))