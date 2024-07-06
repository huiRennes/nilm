"""
Create new train, test and validation datasets from REFIT data.

No normalization is performed, post-process with normalize_dataset.py.
"""

import time
import os
import re
import argparse

import pandas as pd
#import matplotlib.pyplot as plt
current_dir = os.path.dirname(__file__)

DATA_DIRECTORY = os.path.join(current_dir, 'raw_data')
SAVE_DIRECTORY = os.path.join(current_dir, '../../processed_data/refit')

APPLIANCE_NAME = 'washingmachine'

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 4, 9],
        'test_house': [19], #channel 7
        'test_house_channels' : [5],
        'validation_house': [20], # channel 9
        'validation_house_channels' : [9],
        'test_on_train_house': [15], #channel 8
        'test_on_train_house_channels' : [8]
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses':   [2, 3, 5, 6, 8, 9, 12, 13, 15, 18, 19, 20],
        'channels': [5, 8, 7, 6, 8, 6,  3,  8,  7,  9,  4,  8],
        'test_house': [4],
        'test_house_channels' : [8],
        'validation_house': [17],
        'validation_house_channels' : [7],
        'test_on_train_house': [10],
        'test_on_train_house_channels' : [8]
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses':   [1, 2, 3,  7, 9, 17, 20],
        'channels': [1, 1, 2,  1, 1,  2, 1 ],
        'test_house': [15],
        'test_house_channels' : [1],
        'validation_house': [12],
        'validation_house_channels' : [1],
        'test_on_train_house': [5],
        'test_on_train_house_channels' : [1]
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses':   [1, 2, 3, 5, 6, 7, 9, 10, 15, 16],
        'channels': [6, 3, 5, 4, 3, 6, 4, 6,   4,   6],
        'test_house': [20],
        'test_house_channels' : [5],
        'validation_house': [18],
        'validation_house_channels' : [6],
        'test_on_train_house': [13],
        'test_on_train_house_channels' : [4]
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 220,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses':   [1, 2, 3, 6, 7, 9, 10, 13, 15, 16, 17, 19, 20],
        'channels': [5, 2, 6, 2, 5, 3, 5,   3,  3, 5,   4,  2, 4],
        'test_house': [8],
        'test_house_channels' : [4],
        'validation_house': [18],
        'validation_house_channels' : [5],
        'test_on_train_house': [5],
        'test_on_train_house_channels' : [3]
    }
}

def load(path, building, appliance, channel):
    # load csv
    file_name = os.path.join(path, 'House_' + str(building) + '.csv')
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel+2],
                             na_filter=False,
                             parse_dates=True,
                             memory_map=True
                             )

    return single_csv

def compute_stats(df) -> dict:
    """ Given a Series DataFrame compute its statistics. """
    filtered_values = df[df != 0]
    return {
        'mean': filtered_values.mean(),
        'std': filtered_values.std(),
        'median': filtered_values.median(),
        'quartile1': filtered_values.quantile(q=0.25, interpolation='lower'),
        'quartile3': filtered_values.quantile(q=0.75, interpolation='lower')
    }

def main():
    start_time = time.time()
    appliance_name = APPLIANCE_NAME
    print(appliance_name)
    
    path = DATA_DIRECTORY

    save_path = os.path.join(SAVE_DIRECTORY, APPLIANCE_NAME)
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    print(f'data path: {path}')
    print(f'save path: {save_path}')
    
    total_length = 0
    print("Creating datasets...")
    # Looking for proper files
    for _, filename in enumerate(os.listdir(path)):
        if int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['test_house']:
            print('File: ' + filename + ' test set')
            # Loading
            test = load(path,
                 int(re.search(r'\d+', filename).group()),
                 appliance_name,
                 params_appliance[appliance_name]['test_house_channels'][params_appliance[appliance_name]['test_house']
                        .index(int(re.search(r'\d+', filename).group()))]
                 )

            print(test.iloc[:, 0].describe())
            #test.iloc[:, 0].hist()
            #plt.show()
            print(test.iloc[:, 1].describe())
            #test.iloc[:, 1].hist()
            #plt.show()

            agg_stats = compute_stats(test.iloc[:, 0])
            print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
            print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
            app_stats = compute_stats(test.iloc[:, 1])
            print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
            print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
            # Save
            fname = os.path.join(save_path, f'{appliance_name}_test_H{params_appliance[appliance_name]["test_house"][0]}.csv')
            test.to_csv(fname, index=False)
    
            print("Size of test set is {:.3f} M rows (House {:d})."
                  .format(test.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_house'][0]))
            del test
    
        elif int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['validation_house']:
            print('File: ' + filename + ' validation set')
            print(params_appliance[appliance_name]['validation_house_channels'][params_appliance[appliance_name]['validation_house']
                        .index(int(re.search(r'\d+', filename).group()))])
        
            # Loading
            val = load(path,
                 int(re.search(r'\d+', filename).group()),
                 appliance_name,
                 params_appliance[appliance_name]['validation_house_channels']
                 [params_appliance[appliance_name]['validation_house']
                        .index(int(re.search(r'\d+', filename).group()))]
                 )
            
            print(val.iloc[:, 0].describe())
            #val.iloc[:, 0].hist()
            #plt.show()
            print(val.iloc[:, 1].describe())
            #val.iloc[:, 1].hist()
            #plt.show()

            agg_stats = compute_stats(val.iloc[:, 0])
            print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
            print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
            app_stats = compute_stats(val.iloc[:, 1])
            print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
            print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
            # Save
            fname = os.path.join(save_path, f'{appliance_name}_validation_H{params_appliance[appliance_name]["validation_house"][0]}.csv')
            val.to_csv(fname, index=False)
    
            print("Size of validation set is {:.3f} M rows (House {:d})."
                  .format(val.shape[0] / 10 ** 6, params_appliance[appliance_name]['validation_house'][0] ))
            del val
    
        elif int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['houses']:
            print('File: ' + filename)
            print('    House: ' + re.search(r'\d+', filename).group())
    
            # Loading
            try:
                csv = load(path,
                           int(re.search(r'\d+', filename).group()),
                           appliance_name,
                           params_appliance[appliance_name]['channels']
                           [params_appliance[appliance_name]['houses']
                                  .index(int(re.search(r'\d+', filename).group()))]
                           )

                print(csv.iloc[:, 0].describe())
                #csv.iloc[:, 0].hist()
                #plt.show()
                print(csv.iloc[:, 1].describe())
                #csv.iloc[:, 1].hist()
                #plt.show()

                agg_stats = compute_stats(csv.iloc[:, 0])
                print(f'aggregate - mean: {agg_stats["mean"]}, std: {agg_stats["std"]}')
                print(f'aggregate - median: {agg_stats["median"]}, quartile1: {agg_stats["quartile1"]}, quartile3: {agg_stats["quartile3"]}')
                app_stats = compute_stats(csv.iloc[:, 1])
                print(f'{appliance_name} - mean: {app_stats["mean"]}, std: {app_stats["std"]}')
                print(f'{appliance_name} - median: {app_stats["median"]}, quartile1: {app_stats["quartile1"]}, quartile3: {app_stats["quartile3"]}')
    
                rows, _ = csv.shape
                total_length += rows
    
                if filename == 'House_' + str(params_appliance[appliance_name]['test_on_train_house']) + '.csv':
                    fname = os.path.join(
                        save_path,
                        f'{appliance_name}_test_on_train_H{params_appliance[appliance_name]["test_on_train_house"]}.csv')
                    csv.to_csv(fname, index=False)
                    print("Size of test on train set is {:.3f} M rows (House {:d})."
                          .format(csv.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_on_train_house']))
    
                # saving the whole merged file
                fname = os.path.join(save_path, f'{appliance_name}_training_.csv')
                # Append df to csv if it exists with header only the first time.
                csv.to_csv(fname, mode = 'a', index = False, header = not os.path.isfile(fname))
                del csv
    
            except:
                pass
    
    print("Size of training set is {:.3f} M rows.".format(total_length / 10 ** 6))
    print("\nTraining, validation and test sets are  in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()