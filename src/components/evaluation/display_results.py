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

# plt.style.use(['science'])

def visualize_performance(data: pd.DataFrame, results_folder: str):

    # Plot pandas dataframe with performance information
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    data.plot(ax=ax, marker='o', markersize=8)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Information')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance_information.png'))
    

if __name__ == '__main__':
    visualize_performance_information_disclosure()