import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = './output/basic_properties'
GROUPS = [0, 4, 9, 14]
GROUP_NAMES = ["Layer size", "Number layers", "Input order", "Number layers"]

if __name__ == '__main__':
    dirs_data = dict()
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if '.npy' in file:
                path = os.path.join(root, file)
                dirs_data[path] = np.load(path)

    stats = list()
    for directory, data in dirs_data.items():
        dir_split = directory.split('\\')[1:]
        dataset = dir_split[0]
        distance_mat_name = dir_split[1].split('_')[0]
        data_norm = data / np.max(data)
        for idx, experiment_idx in enumerate(GROUPS):
            if idx < (len(GROUPS) - 1):
                data_part = data_norm[experiment_idx:GROUPS[idx + 1], experiment_idx:GROUPS[idx + 1]]
            else:
                data_part = data_norm[experiment_idx:, experiment_idx:]
            data_part = data_part[np.triu_indices_from(data_part, k=1)]
            stats.append({'Dataset': dataset, 'Discretization': distance_mat_name, 'Experiment': GROUP_NAMES[idx],
                          'Mean': np.mean(data_part), 'Standard deviation': np.std(data_part)})
        output_path = f'{os.path.splitext(directory)[0]}.png'
        plt.figure()
        plt.imshow(data)
        plt.xticks(np.arange(data.shape[0]))
        plt.yticks(np.arange(data.shape[1]))
        plt.savefig(output_path)
    pd.DataFrame(stats).to_csv(os.path.join(PATH, 'stats.csv'), index=False)
