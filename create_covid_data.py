#!/home/awikner1/miniconda3/envs/denpool/bin/python
#SBATCH -t 60
#SBATCH -n 1
#SBATCH -c 1
import h5py
import pandas as pd
import numpy as np
import os, glob
from tqdm import tqdm

file_count = 0
data_path = '/scratch/zt1/project/edott-prj/user/awikner1/covid19-forecast-hub/data-processed'
for root_dir, cur_dir, files in os.walk(data_path):
    for data_file in files:
        if '.csv' in data_file or ('metadata' in data_file and '.txt' in data_file):
            file_count += 1
print('Total number of files to read: %d' % file_count)


with h5py.File('covid_prediction_data.hdf5','a') as f:
    # Get all groups and paths to data
    group_names = next(os.walk(data_path))[1]
    group_paths = [os.path.join(data_path, group_name) for group_name in group_names]
    with tqdm(total = file_count) as pbar:
        for group_name, group_path in zip(group_names, group_paths):
            # Create group in file
            group = f.create_group(group_name)
            # Load metadata and save as group attributes
            with open(os.path.join(group_path, 'metadata-%s.txt' % group_name), 'r') as meta_file:
                content = meta_file.read().split(': ')
            content_2 = [entry.split('\n') for entry in content[1:]]
            fields = [content[0]]
            fields.extend([entry[-1] for entry in content_2[:-1]])
            entries = [' '.join(entry[:-1]).replace('  ',' ') for entry in content_2]
            for field, entry in zip(fields, entries):
                group.attrs[field] = entry
            pbar.update(1)
            data_files = glob.glob(os.path.join(group_path, '*.csv'), recursive = True)
            data = pd.read_csv(data_files[0], engine='pyarrow')
            pbar.update(1)
            for data_file in data_files[1:]:
                data = pd.concat((data, pd.read_csv(data_file, engine='pyarrow')),
                    axis = 0, join = "outer", ignore_index = True)
                pbar.update(1)
            data['forecast_date']   = pd.to_datetime(data['forecast_date'])
            data['target_end_date'] = pd.to_datetime(data['target_end_date'])
            data.sort_values(by='forecast_date')
            locations = pd.unique('location')
            targets   = pd.unique('target')
            quantiles = pd.unique('quantile')
            quantiles = quantiles[quantiles != 'NA']
            for location in locations:
                if location not in group:
                    location_grp = group.create_group(location)
                else:
                    location_grp = group[location]
                location_data = data.loc[data['location'] == location]
                for target in targets:
                    if target not in location_grp:
                        target_grp = location_grp.create_group(target)
                    else:
                        target_grp = location_grp[target]
                    target_data = location_data.loc[data['target'] == target]
                    for quantile in quantiles:
                        print(quantile)
                        quantile_data = target_data.loc[data['quantile'] == quantile][[
                                                        'forecast_date',
                                                        'target_end_date',
                                                        'value']]
                        target_grp.create_dataset('quantile%s' % quantile, quantile_data.to_records())
                    point_data = target_data.loc[data['type'] == 'point'][[
                                                        'forecast_date',
                                                        'target_end_date',
                                                        'value']]
                    target_grp.create_dataset('point', point_data.to_records())


                            
