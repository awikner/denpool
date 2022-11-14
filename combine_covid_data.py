import os, h5py, datetime
from tqdm import tqdm

data_path = 'C:\\Users\\user\\Documents\\covid19-forecast-hub\\data-processed'
raw_data_path = 'C:\\Users\\user\\Documents\\covid19-forecast-hub\\data-processed'
denpool_path = 'C:\\Users\\user\\Documents\\covid-data'
#running = ['RobertWalraven-ESG','CEID-Walk','MITCovAlliance-SIR','Google_Harvard-CPF']
#group_names = next(os.walk(data_path))[1]
group_names = ['JHUAPL-Bucky', 'USC-SI_kJalpha']
#rerun_file = os.path.join(denpool_path, 're_run_groups.txt')
#rerun_groups = []

with h5py.File(os.path.join(denpool_path, 'covid_prediction_data.hdf5'), 'a') as f_base:
    for group_name in group_names:
        if os.path.exists(os.path.join(denpool_path, 'covid_prediction_data_%s.hdf5' % group_name)):
            try:
                with h5py.File(os.path.join(denpool_path, 'covid_prediction_data_%s.hdf5' % group_name), 'r') as f:
                    print(group_name)
                    f.copy(group_name, f_base)
                    timestamps = [file[:10] for file in os.listdir(os.path.join(data_path, group_name)) if '.csv' in file]
                    dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps]
                    dates.sort()
                    sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
                    f_base[group_name].attrs['dates'] = sorteddates
            except:
                print('%s did not run properly.' % group_name)
                pass
        else:
            print('%s not found.' % group_name)
