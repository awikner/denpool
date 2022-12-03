from src.classes import *
from src.helpers import *
import h5py
import pandas as pd
import timeit
from matplotlib import pyplot as plt
from datetime import datetime
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.set_flush_denormal(True)

# Covid Data Parameters

truth_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_truth_data.hdf5'
ensemble_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_prediction_data.hdf5'
ensemble_models = ['IHME-CurveFit', 'CU-scenario_mid', 'MOBS-GLEAM_COVID', 'CU-select', 'CU-scenario_low',
                  'CU-nochange', 'GT-DeepCOVID', 'PSI-DRAFT', 'USC-SI_kJalpha']
time_delays = 5
pred_data = h5py.File(ensemble_path, 'r')
all_dates = []
for i in range(len(ensemble_models)):
    all_dates.extend([datetime.strptime(str(elem[1])[2:-1], '%y-%m-%d')\
                for elem in pred_data[ensemble_models[i]]['01']['1_wk_ahead_inc_death']['point'][:]])
dates = np.unique(np.array([date.strftime("%y-%m-%d") for date in sorted(all_dates)])).tolist()
print(len(dates))
print(dates)
# Covid Data Parameters

truth_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_truth_data.hdf5'
ensemble_path = 'C:\\Users\\user\\Documents\\covid-data\\covid_prediction_data.hdf5'
ensemble_models = ['IHME-CurveFit', 'CU-scenario_mid', 'MOBS-GLEAM_COVID', 'CU-select', 'CU-scenario_low',
                  'CU-nochange', 'GT-DeepCOVID', 'PSI-DRAFT', 'USC-SI_kJalpha']
time_delays = 5
pred_data = h5py.File(ensemble_path, 'r')
alphas = ['point', '0.02', '0.05', '0.1', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
prediction_type = ['1_wk_ahead_inc_death', 7]
truth_type = 'Incident Deaths'
extra_truth = ['Incident Cases']
data_start = '20-05-09'
data_end   = '22-11-05'
#train_end  = '20-08-22'
#test_end   = '21-02-20'
train_end  = '21-05-29'
test_end   = '21-10-30'
#train_end  = '21-12-11'
#test_end   = '22-06-18'
#train_end  = '22-07-02'
#test_end   = '22-11-05'
train_start, test_start = get_start_dates(dates, train_end, test_end, time_delays)
train_start_end = [(data_start, train_end), (train_start, data_end)]
test_start_end  = [(test_start, test_end)]
#train_end_date  = '22-07-02'
#test_start_date = '22-07-02'
#test_end_date   = '22-11-05'
train_locations =  ['01', '02', '04', '05', '06', '08', '09', '10','11', '12', '13', '16', '17',
              '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
              '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
              '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56']
test_locations = ['01', '02', '04', '05', '06', '08', '09', '10','11', '12', '13', '16', '17',
              '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
              '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
              '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56']

#ensemble_data = CovidDataAllTimes(truth_path, ensemble_path, ensemble_models, alphas, prediction_type,
#                 truth_type,train_start_end, test_start_end, train_locations, test_locations, extra_truth = extra_truth)

#time_delay = 5
#covid_data = CovidDataLoader(ensemble_data, time_delay, batch_size = 64, dtype = dtype, use_wis_keys=True)
#torch.save(covid_data, 'covid_dataloader_td5_wcase_wwis_start_%s_end_%s.pth' % (train_start_end[0][1], test_start_end[0][1]))
#covid_data = torch.load('covid_dataloader_td5.pth')

#covid_data = CovidDataAllTimes(truth_path, ensemble_path, ensemble_models, alphas, prediction_type,
#                 truth_type,train_start_end, test_start_end, train_locations, test_locations)

#print(covid_data.train_model_data.shape)
#print(covid_data.test_model_data.shape)

#time_delay = 5
#covid_data = CovidDataLoader(covid_data, time_delay, batch_size = 64, dtype = dtype)

@d2l.add_to_class(CovidAdditiveAttention)
def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr, weight_decay = self.weight_decay)
#@d2l.add_to_class(CovidAdditiveAttention)
#def loss(self, y_hat, y):
#    return torch.log(weighted_interval_score(y_hat, y, self.alphas)).mean()
covid_data = torch.load('covid_dataloader_td5_wcase_wwis_start_%s_end_%s.pth' % (train_start_end[0][1], test_start_end[0][1]))
time_delay = 5
#dtype = torch.float64
#torch.set_default_dtype(dtype)
output_size = 1+2*(len(alphas)-1)
query_size, key_size, num_hidden, dropout, lr, epochs = time_delay*2, (output_size+1)*time_delay, 1000, 0.1, 5e-5, 201
alphas_eval = torch.tensor([float(alpha) for alpha in covid_data.alphas_eval[1:]]).reshape(1,1,-1)
attention   = CovidAdditiveAttention(query_size, key_size, num_hidden, dropout, lr, alphas_eval, l1_reg = 0.)
attention.device = "cpu"
trainer = CovidTrainer(max_epochs = epochs)
attention.weight_decay = 1e-6

trainer.fit(attention, covid_data)