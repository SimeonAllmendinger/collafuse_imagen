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
    ax.set_xlabel(r'$t_{\zeta}$')
    ax.set_ylabel('Fréchet Clip Distance')
    ax.set_title(r'Image fidelity for different values of $t_{\zeta}$')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'performance.png'))
    
def visualize_all_performances(results_folder: str) -> None:

    result_files = sorted(glob.glob(os.path.join(results_folder, '*.csv')))
    
    # Plot pandas dataframe with performance information
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    gen_list = list()
    cloud_cut_list = list()
    client_from_noise_list = list()
    cloud_approx_list = list()
    cut_list = list()

    for i, file in enumerate(result_files):
        df = pd.read_csv(file)
        df['t_cut'] = pd.to_numeric(df['t_cut'], errors='coerce')
        df['clip_fid'] = pd.to_numeric(df['clip_fid'], errors='coerce')
        
        df_gen = df[(df['client_id'] != 'ALL') & (df['dir_name'] == 'generated/')]
        df_cloud_cut = df[(df['client_id'] != 'ALL') & (df['dir_name'] == 'cloud_cut/')]
        df_client_from_noise = df[(df['client_id'] != 'ALL') & (df['dir_name'] == 'client_from_noise/')]
        df_cloud_approx = df[(df['client_id'] != 'ALL') & (df['dir_name'] == 'cloud_approx/')]
        df_cut = df[(df['client_id'] == 'ALL') & (df['dir_name'] == 'cut/')]
        
        gen_list.append(df_gen['clip_fid'].mean())
        cloud_cut_list.append(df_cloud_cut['clip_fid'].mean())
        client_from_noise_list.append(df_client_from_noise['clip_fid'].mean())
        cloud_approx_list.append(df_cloud_approx['clip_fid'].mean())
        cut_list.append(df_cut['clip_fid'].mean())

        ax.scatter(df_gen['t_cut'], df_gen['clip_fid'], marker='^', color='red', label=r'$\hat{x}^c_0 \circ \hat{x}^s_{t_\zeta}(\epsilon)$' if i == 0 else None)
        ax.scatter(df_cloud_cut['t_cut'], df_cloud_cut['clip_fid'], marker='^', color='blue', label=r'$\hat{x}^s_{t_\zeta}(\epsilon)$' if i == 0 else None)
        ax.scatter(df_client_from_noise['t_cut'], df_client_from_noise['clip_fid'], marker='^', color='green', label=r'$\hat{x}^c_0(\epsilon)$' if i == 0 else None)
        ax.scatter(df_cloud_approx['t_cut'], df_cloud_approx['clip_fid'], marker='^', color='purple', label=r'$\hat{x}^s_0(\epsilon)$' if i == 0 else None)
        ax.scatter(df_cut['t_cut'], df_cut['clip_fid'], marker='^', color='orange', label=r'$x_{t_\zeta}$' if i == 0 else None)
    
    t_cuts = np.linspace(0, 1000, len(result_files),endpoint=True, dtype=int)
    ax.plot(t_cuts,gen_list,color='red')
    ax.plot(t_cuts,cloud_cut_list,color='blue')
    ax.plot(t_cuts,client_from_noise_list,color='green')
    ax.plot(t_cuts,cloud_approx_list,color='purple')
    ax.plot(t_cuts,cut_list,color='orange')

    ax.set_xlabel(r'$t_{\zeta}$')
    ax.set_ylabel('Fréchet Clip Distance (FCD)')
    ax.set_title(r'Image fidelity for different values of $t_{\zeta}$')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join('./performance.png'))
    

if __name__ == '__main__':
    visualize_performance_information_disclosure()