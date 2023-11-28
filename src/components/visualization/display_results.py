import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import os
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

def visualize_performance_information_disclosure():
    
    # load data from kid folder
    kid_folder = "./results/kid"
    mse_folder='results/pixel_comparison_mse'
    

    # Get the list of files in the kid folder
    kid_p_train_files = sorted(glob.glob(os.path.join(kid_folder, '*performance_on_train*.csv')))
    kid_p_test_files = sorted(glob.glob(os.path.join(kid_folder, '*performance_on_test*.csv')))
    kid_id_train_files = sorted(glob.glob(os.path.join(kid_folder, '*disclosure_on_train*.csv')))
    kid_id_test_files = sorted(glob.glob(os.path.join(kid_folder, '*disclosure_on_test*.csv')))
    pixel_id_train_files = sorted(glob.glob(os.path.join(mse_folder, '*.csv')))

    # Load the data from the kid folder
    fig = plt.figure(figsize=(10, 5))
    
    ax11 = fig.add_subplot(221)  # add subplot into first position in a 2x2 grid (upper left)
    ax12 = fig.add_subplot(223, sharex=ax11)  # add to third position in 2x2 grid (lower left) and sharex with ax11
    ax13 = fig.add_subplot(222)  # add subplot to cover both upper and lower right, in a 2x2 grid. This is the same as the rightmost panel in a 1x2 grid.
    ax14 = fig.add_subplot(224, sharex=ax13)  # add subplot 
    
    
    bottom_train=0
    bottom_test=0
    bottom_id_train=0
    bottom_id_pixel=0
    
    for i, (kid_p_train_file, kid_p_test_file, kid_id_train_file, kid_id_test_file, pixel_id_train_file) in enumerate(zip(kid_p_train_files,kid_p_test_files,kid_id_train_files,kid_id_test_files,pixel_id_train_files)):
        
        # Load the data with pandas read_csv
        kid_p_train_data = pd.read_csv(kid_p_train_file)
        kid_p_test_data = pd.read_csv(kid_p_test_file)
        kid_id_train_data = pd.read_csv(kid_id_train_file)
        kid_id_test_data = pd.read_csv(kid_id_test_file)
        pixel_id_train_data = pd.read_csv(pixel_id_train_file)

        # Use the loaded data for visualization or further processing
        
        ax11.bar(np.linspace(0,5,6),kid_p_train_data.iloc[:,1::3].values[0][-1:5:-1], bottom=bottom_train, color=f'C{i}')
        ax12.bar(np.linspace(0,5,6),kid_p_test_data.iloc[:,1::3].values[0][-1:5:-1], bottom=bottom_test, color=f'C{i}')
        ax13.bar(np.linspace(0,5,6),kid_id_train_data.iloc[:,1::3].values[0][-1:5:-1], bottom=bottom_id_train, color=f'C{i}')
        ax14.bar(np.linspace(0,5,6),pixel_id_train_data.iloc[:,1::3].values[0][-1::-1], bottom=bottom_id_pixel, color=f'C{i}')
        
        bottom_train=bottom_train+kid_p_train_data.iloc[:,1::3].values[0][-1:5:-1]
        bottom_test=bottom_test+kid_p_test_data.iloc[:,1::3].values[0][-1:5:-1]
        bottom_id_train=bottom_id_train+kid_id_train_data.iloc[:,1::3].values[0][-1:5:-1]
        bottom_id_pixel=bottom_id_pixel+pixel_id_train_data.iloc[:,1::3].values[0][-1::-1]
        
    ax11.set_xticks([0,1,2,3,4,5])
    ax12.set_xticks([0,1,2,3,4,5])
    
    ax11.set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    ax12.set_xlabel('Percentage of local training (cut-ratio)')
    ax11.set_ylabel('KID score on client data')
    ax12.set_ylabel('KID score on test data')
    ax11.set_title('PERFORMANCE')
    
    ax13.set_xticks([0,1,2,3,4,5])
    ax13.set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    ax13.set_ylabel('KID score on client data')
    ax13.set_title('INFORMATION DISCLOSURE')
    
    ax13.set_xticks([0,1,2,3,4,5])
    ax13.set_xticklabels(['0%','20%','40%','60%','80%','100%'])
    ax14.set_xlabel('Percentage of local training (cut-ratio)')
    ax13.set_ylabel('KID score on client data')
    ax13.set_title('DISCLOSED INFORMATION')
    ax14.set_ylabel('Mean MSE on client data')
    
    plt.legend(['Client 1', 'Client 2', 'Client 3'],loc='best')
    fig.savefig('./kid_performance_information_disclosure.png')
    fig.savefig('./kid_performance_information_disclosure.pdf')
    fig.savefig('./kid_performance_information_disclosure.svg')

def visualize_resources():
    
    df_gpu_process_utilization = pd.read_csv('./results/resources/Process_GPU_Utilization_Percentage.csv')
    
    fig, ax = plt.subplots()
    ax.plot(df_gpu_process_utilization)
    fig.savefig('./Process_GPU_Utilization_Percentage.png')

if __name__ == '__main__':
    visualize_performance_information_disclosure()