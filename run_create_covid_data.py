import os

data_path = '/scratch/zt1/project/edott-prj/user/awikner1/covid19-forecast-hub/data-processed'
denpool_path = '/scratch/zt1/project/edott-prj/user/awikner1/denpool/'
if not os.path.exists(os.path.join(denpool_path, 'data')):
    os.mkdir(os.path.join(denpool_path, 'data'))
#with open(os.path.join(denpool_path, 're_run_groups.txt'), 'r') as f:
#    lines = f.readlines()
#group_names = [line.replace('\n','') for line in lines]
group_names = ['JHUAPL-Bucky']
group_paths = [os.path.join(data_path, group_name) for group_name in group_names]
script_path = '/scratch/zt1/project/edott-prj/user/awikner1/denpool/create_covid_data.py'
save_path   = os.path.join(denpool_path, 'data')

for iter, (group_name, group_path) in enumerate(zip(group_names, group_paths)):
    if os.path.exists(os.path.join(os.path.join(denpool_path, 'data'), 'covid_prediction_data_%s.hdf5' % group_name)):
        os.remove(os.path.join(os.path.join(denpool_path, 'data'), 'covid_prediction_data_%s.hdf5' % group_name))
    submit_str = 'sbatch %s -G %s -P %s -S %s' % (script_path, group_name, group_path, save_path)
    print('Group %d/%d: %s' % (iter+1, len(group_names), group_name))
    os.system(submit_str)
