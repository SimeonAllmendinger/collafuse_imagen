import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

import os
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

def visualize_kid_performance():
    
    # load data from kid folder
    kid_folder = "./results/kid"

    # Get the list of files in the kid folder
    kid_train_files = sorted(glob.glob(os.path.join(kid_folder, '*performance_on_train*.csv')))
    kid_test_files = sorted(glob.glob(os.path.join(kid_folder, '*performance_on_test*.csv')))

    # Load the data from the kid folder
    data = []
    fig, ax = plt.subplots()
    
    for i, (kid_train_file, kid_test_file) in enumerate(zip(kid_train_files,kid_test_files)):
        
        # Load the data with pandas read_csv
        kid_train_data = pd.read_csv(kid_train_file)
        kid_test_data = pd.read_csv(kid_test_file)

        # Use the loaded data for visualization or further processing
        #ax.plot(kid_train_data.iloc[:,1::3].values[0][:6], linestyle='--', label=f'Client {1+i} stacked')
        ax.plot(kid_train_data.iloc[:,1::3].values[0][6:], linestyle=':', label=f'Client {1+i} (train dataset)', color=f'C{i}')
        ax.plot(kid_test_data.iloc[:,1::3].values[0][6:], linestyle='-', label=f'Client {1+i} (hold-out dataset)', color=f'C{i}')
    
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    ax.set_xlabel('Percentage of local training')
    ax.set_ylabel('KID score')
    ax.set_title('KID score on train and test set')
    plt.legend(loc='upper right')
    fig.savefig('./kid_performance_on_train.png')

if __name__ == '__main__':
    visualize_kid_performance()