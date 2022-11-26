import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from natsort import natsorted
import seaborn as sns
import numpy as np

from trackobjects.simulationconstants import SimulationConstants
from trackobjects.symmetricmerge import SymmetricMergingTrack

path_to_data_csv = os.path.join('..', 'data_folder', 'crt_who_is_first_exit.csv')
global_crt_index = pd.read_csv(path_to_data_csv, sep=',')

fig, ax1 = plt.subplots()

sns.histplot(round(global_crt_index['crt_index_50_50']), bins=20)
sns.histplot(round(global_crt_index['crt_index_50_50']), bins=20)
sns.histplot(round(global_crt_index['crt_index_50_50']), bins=20)

plt.show()
