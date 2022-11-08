#!/home/awikner1/miniconda3/envs/denpool/bin/python -u
#SBATCH -t 240
#SBATCH -n 1
#SBATCH -c 1
import h5py
import pandas as pd
import numpy as np
import os, glob, sys
from tqdm import tqdm
import getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "G:P:S:",[])
    except getopt.GetoptError:
        print('Error: Some options not recognized')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-G':
            group_name = str(arg)
            print('Group Name: %s' % group_name)
        elif opt == '-P':
            group_path = str(arg)
            print('Group Path: %s' % group_path)
        elif opt == '-S':
            save_path = str(arg)
            print('Save Path: %s' % save_path)

    file_path = os.path.join(save_path, 'covid_prediction_data_%s.hdf5' % group_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    with h5py.File(file_path,'a') as f:
        # Get all groups and paths to data
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
        #pbar.update(1)
        data_files = glob.glob(os.path.join(group_path, '*.csv'), recursive = True)
        print('Loading data...')
        with tqdm(total = len(data_files)) as pbar:
            data = pd.read_csv(data_files[0], engine='pyarrow')
            pbar.update(1)
            for data_file in data_files[1:]:
                data = pd.concat((data, pd.read_csv(data_file, engine='pyarrow')),
                    axis = 0, join = "outer", ignore_index = True)
                pbar.update(1)
        data['forecast_date']   = pd.to_datetime(data['forecast_date'])
        data['target_end_date'] = pd.to_datetime(data['target_end_date'])
        data.sort_values(by='forecast_date')
        data['forecast_date']   = data['forecast_date'].dt.strftime('%y-%m-%d')
        data['target_end_date'] = data['target_end_date'].dt.strftime('%y-%m-%d')
        data['target'] = data['target'].replace(' ','_',regex=True)
        data['location'] = data['location'].astype('str')
        locations = np.sort(pd.unique(data['location']))
        targets   = pd.unique(data['target'])
        quantiles = pd.unique(data['quantile'])
        #print(quantiles)
        if np.all(quantiles == None):
            quantiles = []
        else:
            quantiles = quantiles[np.logical_not(np.isnan(quantiles))]
        #print(locations)
        #print(targets)
        #print(quantiles)
        #print(locations)
        #print(targets)
        #print(quantiles)
        print('Saving data...')
        if 'point' in pd.unique(data['type']):
            #print('Using point')
            total_datasets = len(locations)*len(targets)*(len(quantiles)+1)
        else:
            total_datasets = len(locations) * len(targets) * (len(quantiles))
        with tqdm(total = total_datasets) as pbar:
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
                    target_data = location_data.loc[location_data['target'] == target]
                    for quantile in quantiles:
                        #print(quantile)
                        quantile_data = target_data.loc[target_data['quantile'] == quantile][[
                                                        'forecast_date',
                                                        'target_end_date',
                                                        'value']]
                        quantile_rec = quantile_data.to_records(index=False)
                        #print(quantile_rec)
                        quantile_arr = quantile_rec.view(quantile_rec.dtype.fields or quantile_rec.dtype, np.ndarray)
                        #print(quantile_arr)
                        qdt = [(name, 'S8') if dtype == '|O' else (name, dtype) for name, dtype in quantile_arr.dtype.descr]
                        #print(qdt)
                        quantile_arr = quantile_arr.astype(qdt)
                        #print(quantile_arr)
                        #print(targets)
                        #print(group_name, location, target, quantile)
                        target_grp.create_dataset('quantile%s' % quantile, data = quantile_arr)
                        pbar.update(1)
                    if 'point' in pd.unique(data['type']):
                        #print('Saving point data...')
                        point_data = target_data.loc[target_data['type'] == 'point'][[
                                                            'forecast_date',
                                                            'target_end_date',
                                                            'value']]
                        point_rec = point_data.to_records(index=False)
                        point_arr = point_rec.view(point_rec.dtype.fields or point_rec.dtype, np.ndarray)
                        qdt = [(name, 'S8') if dtype == '|O' else (name, dtype) for name, dtype in point_arr.dtype.descr]
                        #print(qdt)
                        point_arr = point_arr.astype(qdt)
                        #print('Point data size: %d' % point_data.shape[0])
                        target_grp.create_dataset('point', data = point_arr)
                        pbar.update(1)

if __name__ == "__main__":
    main(sys.argv[1:])
