import json
import matplotlib.pyplot as plt
import glob
import os

def show_curve(json_file):
    with open(json_file) as f:
        data = json.load(f)
        scores_per_horizon = data['scores_per_horizon']
        i_list = []
        nrmse_values_list = []
        for i, nrmse in scores_per_horizon.items():
            for nrmse_keys, nrmse_values in nrmse.items():
                i_list.append(i)
                nrmse_values_list.append(nrmse_values)
        # print(i_list, nrmse_values_list)
        max_value = max(nrmse_values_list)

        plt.figure(figsize=(15, 10))
        plt.title('Training NRMSE')
        plt.ylabel('NRMSE')
        plt.xlabel('i')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(i_list, nrmse_values_list)
        plt.show()

result1 = show_curve(r'D:\OneDrive\Captures\Masterarbeit\code_of_masterarbeit\system_identification_of_ship\results\repeat-1\LinearLag-5\scores-test-w_60-h_60.json')

# def show_histogram():

def traverse_folder_with_json(folder_path, json):
    folder_list = os.listdir(folder_path)
    file_list = []
    max_value_list = []
    for file in folder_list:
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            pattern = os.path.join(file_path, '*.' + 'json')
            file_lists = glob.glob(pattern)
            for file_path in file_lists:
                # print(file_path)
                with open(file_path) as f:
                    data = json.load(f)
                    scores_per_horizon = data['scores_per_horizon']
                    i_list = []
                    nrmse_values_list = []
                    for i, nrmse in scores_per_horizon.items():
                        for nrmse_keys, nrmse_values in nrmse.items():
                            i_list.append(i)
                            nrmse_values_list.append(nrmse_values)
                    # print(i_list, nrmse_values_list)
                    max_value = max(nrmse_values)
        # print(file, max_value)
        file_list.append(file)
        max_value_list.append(max_value)
    print(file_list, max_value_list)
    plt.bar(file_list, max_value_list)
    plt.show()




traverse_folder_with_json(r'D:\OneDrive\Captures\Masterarbeit\code_of_masterarbeit\system_identification_of_ship\results\repeat-1', json)