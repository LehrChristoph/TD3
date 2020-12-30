import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def import_data(import_folder):
    data = {}

    for root, dirs, files in os.walk(import_folder):
        for file_name in files:
            filepath = os.path.join(root, file_name)
            print(filepath)
            with open(filepath, "rb") as fd:
                data_file = np.load(fd, allow_pickle=True)
                data[file_name[:-4]] = data_file
            # end with
        # end for
    # end for

    return data
# end def

def generate_figures(datasets, export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    # end if

    for dataset_name in datasets:
        plot_convergence(dataset_name, datasets[dataset_name], export_folder)
    # end for
# end def

def  plot_convergence(dataset_name, dataset, export_folder):
    eval_iter = 5000
    x_axis =  np.arange(0, eval_iter*len(dataset), eval_iter)
    plt.plot(x_axis, dataset)
    
    plt.title(dataset_name)
    
    plt.ylabel('Reward')
    plt.xlabel('Timesteps')
    plt.xlim(min(x_axis), min(max(x_axis), 200000))
    plt.ylim(0, 1010)
    
    filepath = os.path.join(export_folder, dataset_name+'.jpeg')
    plt.savefig(filepath)
    plt.close()
# end def

if __name__ == "__main__":
    datasets = import_data("./results")
    generate_figures(datasets, "./figures")