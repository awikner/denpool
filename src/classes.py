import numpy as np
from numba import int32, float64
from numba.experimental import jitclass
from d2l import torch as d2l
import torch
from src.helpers import median_confidence, weighted_interval_score, interval_score
from scipy.optimize import lsq_linear, minimize, LinearConstraint
import pandas as pd
import time
import timeit
import h5py
from statistics import median
from datetime import datetime
import itertools
from itertools import product

spec = [
    ('tau', float64),
    ('int_steps', int32),
    ('h', float64),
    ('sigma', float64),
    ('beta', float64),
    ('rho', float64),
    ('state', float64[:]),
]

@jitclass(spec)
class LorenzModel():
    def __init__(self, tau=0.1, int_steps=10, sigma=10.,
                 beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T, discard_len=0):
        model_output = np.zeros((T + discard_len + 1, 3))
        model_output[0] = self.state
        for i in range(T + discard_len):
            model_output[i + 1] = self.forward()

        return model_output[discard_len:]

    def run_array(self, ic_array):
        model_output = np.zeros(ic_array.shape)
        for j in range(ic_array.shape[0]):
            self.state = ic_array[j]
            model_output[j] = self.forward()
        return model_output

    def forward(self):
        for i in range(self.int_steps):
            self.state = self.rk4()
        return self.state

    def rk4(self):
        # Fourth order Runge-Kutta integrator
        x = self.state
        k1 = self.dxdt(x)
        k2 = self.dxdt(x + k1 / 2 * self.h)
        k3 = self.dxdt(x + k2 / 2 * self.h)
        k4 = self.dxdt(x + self.h * k3)

        xnext = x + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xnext

    def dxdt(self, x):
        return np.array([self.sigma * (- x[0] + x[1]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])


spec.extend([('time', float64), ('period', float64)])
@jitclass(spec)
class LorenzModelPeriodicRho():
    def __init__(self, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.time = time
        self.period = period
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError

    def run(self, T, discard_len=0):
        model_output = np.zeros((T + discard_len + 1, 3))
        times = np.zeros(T + discard_len + 1)
        model_output[0] = self.state
        times[0] = self.time
        for i in range(T + discard_len):
            model_output[i + 1], times[i+1] = self.forward()

        return model_output[discard_len:], times[discard_len:]

    def forward(self):
        for i in range(self.int_steps):
            self.state, self.time = self.rk4()
        return self.state, self.time

    def rk4(self):
        # Fourth order Runge-Kutta integrator
        x = self.state
        k1 = self.dxdt(x, self.time)
        k2 = self.dxdt(x + k1 / 2 * self.h, self.time + self.h / 2)
        k3 = self.dxdt(x + k2 / 2 * self.h, self.time + self.h / 2)
        k4 = self.dxdt(x + self.h * k3, self.time + self.h)

        xnext = x + 1 / 6 * self.h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xnext, self.time + self.h

    def dxdt(self, x, t):
        self.rhofun(t)
        return np.array([self.sigma * (- x[0] + x[1]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])
    def rhofun(self, t):
        self.rho = 10 * np.sin(2 * np.pi * t / self.period + 1.5 * np.pi) + 38

class NumpyModel(d2l.Module):
    def __init__(self, numpy_model,dtype):
        super().__init__()
        if not hasattr(numpy_model, 'state'):
            raise ValueError
        self.numpy_model = numpy_model
        self.dtype = dtype

    def forward(self, X):
        #print(X)
        self.numpy_model.state = np.double(X.detach().numpy())
        self.numpy_model.forward()
        return torch.from_numpy(self.numpy_model.state).type(self.dtype)

    def run_array(self, X):
        #print(X)
        numpy_states = np.double(X.detach().numpy())
        return torch.from_numpy(self.numpy_model.run_array(numpy_states)).type(self.dtype)

    def run(self, x, T):
        self.numpy_model.state = np.double(x.detach().numpy())
        return torch.from_numpy(self.numpy_model.run(T)).type(self.dtype)

class LorenzDataModule(d2l.DataModule):
    def __init__(self, batch_size = 32, val_size = 64):
        super().__init__()
        self.val_size = val_size

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        if train:
            return torch.utils.data.DataLoader(dataset, self.batch_size,
                                               shuffle=train)
        else:
            return torch.utils.data.DataLoader(dataset, self.val_size,
                                               shuffle=train,
                                               sampler=torch.utils.data.SequentialSampler(dataset))

class ProgressBoardVT(d2l.ProgressBoard):
    """Plot data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=[None, None], ylabel=[None, None], xlim=[None, None],
                 ylim=[None, None], xscale=['linear', 'linear'], yscale=['log', 'linear'],
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 legend_loc = [None,None], use_jupyter = True,
                 fig=None, axes=None, figsize=(12, 6), display=True):
        super().__init__()
        self.save_hyperparameters()

    def draw(self, x, y, label, train = True, every_n=1, label_val = None):
        """Defined in :numref:`sec_utils`"""
        Point = d2l.collections.namedtuple('Point', ['x', 'y'])
        if train:
            axis_idx = 0
        else:
            axis_idx = 1
        if not hasattr(self, 'raw_points'):
            self.raw_points = d2l.collections.OrderedDict()
            self.data = d2l.collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = [[], axis_idx]
            self.data[label] = [[], axis_idx, label_val]
        if train:
            points = self.raw_points[label][0]
            line = self.data[label][0]
            points.append(Point(x, y))
            if len(points) != every_n:
                return
            mean = lambda x: sum(x) / len(x)
            line.append(Point(mean([p.x for p in points]),
                              mean([p.y for p in points])))
            points.clear()
        else:
            self.data[label][0] = [Point(xi, yi) for xi, yi in zip(x,y)]
            self.data[label][2] = label_val
        if not self.display:
            return
        if self.use_jupyter:
            d2l.use_svg_display()
        if self.fig is None:
            self.fig, self.axes = d2l.plt.subplots(1, 2, figsize=self.figsize)
        axes = self.axes
        plt_lines, labels = [[], []], [[], []]
        for idx in range(2):
            axes[idx].cla()
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines[v[1]].append(axes[v[1]].plot([p.x for p in v[0]], [p.y for p in v[0]],
                                          linestyle=ls, color=color)[0])
            if v[2]:
                labels[v[1]].append(k + r' = $%0.2f \pm %0.2f$' % (v[2][0], v[2][1]))
            else:
                labels[v[1]].append(k)
        for idx in range(2):
            # if self.axes else d2l.plt.gca()
            if self.xlim[idx]:
                axes[idx].set_xlim(self.xlim[idx])
            if self.ylim[idx]:
                axes[idx].set_ylim(self.ylim[idx])
            if not self.xlabel[idx]:
                if train:
                    self.xlabel[idx] = self.x
                else:
                    self.xlabel[idx] = ''
            axes[idx].set_xlabel(self.xlabel[idx])
            axes[idx].set_ylabel(self.ylabel[idx])
            axes[idx].set_xscale(self.xscale[idx])
            axes[idx].set_yscale(self.yscale[idx])
            if self.legend_loc[idx]:
                axes[idx].legend(plt_lines[idx], labels[idx], loc = self.legend_loc[idx])
            else:
                axes[idx].legend(plt_lines[idx], labels[idx])
        d2l.display.display(self.fig)
        d2l.display.clear_output(wait=True)

class LorenzPeriodicRhoData(LorenzDataModule):
    def __init__(self, true_model, model_zoo, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0, val_size = 64):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val + 1
        self.model_zoo = [NumpyModel(model) for model in model_zoo]
        self.true_model = true_model
        data, times = true_model.run(n, num_discard)
        self.times = times
        model_data = np.zeros((data.shape[0]-1, len(model_zoo), data.shape[1]))
        for j, model in enumerate(model_zoo):
            model_data[:,j] = model.run_array(data[:-1])
        self.values  = torch.from_numpy(model_data[1:])
        self.queries = torch.from_numpy(data[1:-1]).unsqueeze(1)
        self.keys    = torch.from_numpy(model_data[:-1]) - self.queries
        self.y       = torch.from_numpy(data[2:]).unsqueeze(1)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)

class LorenzPeriodicRhoData(LorenzDataModule):
    def __init__(self, true_model, model_zoo, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0, val_size = 64):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val + 1
        self.model_zoo = [NumpyModel(model) for model in model_zoo]
        self.true_model = true_model
        data, times = true_model.run(n, num_discard)
        self.times = times
        model_data = np.zeros((data.shape[0]-1, len(model_zoo), data.shape[1]))
        for j, model in enumerate(model_zoo):
            model_data[:,j] = model.run_array(data[:-1])
        self.values  = torch.from_numpy(model_data[1:])
        self.queries = torch.from_numpy(data[1:-1]).unsqueeze(1)
        self.keys    = torch.from_numpy(model_data[:-1]) - self.queries
        self.y       = torch.from_numpy(data[2:]).unsqueeze(1)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)

class LorenzPeriodicRhoTDEData(LorenzDataModule):
    def __init__(self, true_model, model_zoo, time_delay = 1, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0, val_size = 64, dtype = torch.float64):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val + self.time_delay
        self.model_zoo = [NumpyModel(model, dtype) for model in model_zoo]
        self.true_model = true_model
        data, times = true_model.run(n, num_discard)
        self.times = times[self.time_delay:-1]
        model_data = np.zeros((data.shape[0]-1, len(model_zoo), data.shape[1]))
        for j, model in enumerate(model_zoo):
            model_data[:,j] = model.run_array(data[:-1])
        self.values  = torch.from_numpy(model_data[self.time_delay:])
        #self.queries = torch.from_numpy(data[1:-1]).unsqueeze(1)
        #self.keys    = torch.from_numpy(model_data[:-1]) - self.queries
        self.y       = torch.from_numpy(data[1+self.time_delay:]).unsqueeze(1)
        self.keys = torch.zeros(model_data.shape[0] - self.time_delay,
                                   model_data.shape[1],
                                   model_data.shape[2] * self.time_delay)
        self.queries    = torch.zeros(data.shape[0] - self.time_delay - 1,
                                1,
                                data.shape[1] * self.time_delay)
        for delay in range(self.time_delay):
            self.queries[:,0,delay*data.shape[1]:(delay+1)*data.shape[1]] =\
                torch.from_numpy(data[(self.time_delay-delay):(-1-delay)])
            self.keys[:,:,delay*data.shape[1]:(delay+1)*data.shape[1]] = \
                torch.from_numpy(model_data[(self.time_delay-delay-1):(-1-delay)])
        self.keys = self.keys - self.queries
        self.values  = self.values.type(dtype)
        self.keys    = self.keys.type(dtype)
        self.queries = self.queries.type(dtype)
        self.y       = self.y.type(dtype)


    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)

class LorenzPeriodicRhoTDENoDiffData(LorenzDataModule):
    def __init__(self, true_model, model_zoo, time_delay = 1, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0, val_size = 64, dtype = torch.float64):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val + self.time_delay
        self.model_zoo = [NumpyModel(model, dtype) for model in model_zoo]
        self.true_model = true_model
        data, times = true_model.run(n, num_discard)
        self.times = times[self.time_delay:-1]
        model_data = np.zeros((data.shape[0]-1, len(model_zoo), data.shape[1]))
        for j, model in enumerate(model_zoo):
            model_data[:,j] = model.run_array(data[:-1])
        self.values  = torch.from_numpy(model_data[self.time_delay:])
        #self.queries = torch.from_numpy(data[1:-1]).unsqueeze(1)
        #self.keys    = torch.from_numpy(model_data[:-1]) - self.queries
        self.y       = torch.from_numpy(data[1+self.time_delay:]).unsqueeze(1)
        self.keys = torch.zeros(model_data.shape[0] - self.time_delay,
                                   model_data.shape[1],
                                   model_data.shape[2] * self.time_delay)
        self.queries    = torch.zeros(data.shape[0] - self.time_delay - 1,
                                1,
                                data.shape[1] * self.time_delay)
        for delay in range(self.time_delay):
            self.queries[:,0,delay*data.shape[1]:(delay+1)*data.shape[1]] =\
                torch.from_numpy(data[(self.time_delay-delay):(-1-delay)])
            self.keys[:,:,delay*data.shape[1]:(delay+1)*data.shape[1]] = \
                torch.from_numpy(model_data[(self.time_delay-delay-1):(-1-delay)])
        self.values  = self.values.type(dtype)
        self.keys    = self.keys.type(dtype)
        self.queries = self.queries.type(dtype)
        self.y       = self.y.type(dtype)


    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)


class CovidDataAllTimes(d2l.DataModule):
    def __init__(self, truth_path, ensemble_path, ensemble_models, alphas_eval, prediction_type,
                 truth_type, train_start_end_dates, test_start_end_dates, train_locations, test_locations,
                 rescale=None, interp_max = 2):
        super().__init__()
        self.save_hyperparameters()
        self.locations = sorted(list(set(train_locations).union(set(test_locations))))
        print(len(self.locations))
        self.get_date_range()
        model_data, y = self.download()
        model_data = self.fillin_lin_interp_mean_persistence(model_data)
        if rescale == "inv_max":
            self.rescale_inv_max(y)
        self.train_test_split(model_data, y)

    def get_date_range(self):
        all_dates = []
        for pair in self.train_start_end_dates:
            for date in pair:
                all_dates.append(datetime.strptime(date, '%y-%m-%d'))
        for pair in self.test_start_end_dates:
            for date in pair:
                all_dates.append(datetime.strptime(date, '%y-%m-%d'))
        sorted_dates = [date.strftime("%y-%m-%d") for date in sorted(all_dates)]
        self.date_range = (sorted_dates[0], sorted_dates[-1])
        return

    def download(self):
        """
        Read in truth and ensemble models data. Create numpy arrays to store data.
        self.dates contains [num locations * num dates, 1] array of dates.
        y contains [num locations * num dates, 1] array of ground truth data.
        model_data contains [num locations * num dates, num models, num alphas] array of ensemble models data.
        """
        # Get data handles
        self.f_truth = h5py.File(self.truth_path, 'r')
        self.f_ensemble = h5py.File(self.ensemble_path, 'r')
        # Get target data and dates
        step = self.prediction_type[1]
        self.train_dates_len = []
        self.test_dates_len  = []
        for i, location in enumerate(self.locations):
            # Load in truth data from full data range, get all dates for true data
            targets = self.f_truth[location]['Cumulative ' + self.truth_type.split(' ')[1]][:]
            targets_df = pd.DataFrame(targets['value'],
                                      index=[target.decode('UTF-8') for target in targets['date']])
            if 'Incident' in self.truth_type:
                targets_df = pd.DataFrame(targets_df.subtract(targets_df.shift(7, fill_value=0.)),
                                          index=[target.decode('UTF-8') for target in targets['date']])
            dates_temp = targets_df[(targets_df.index >= self.date_range[0]) & \
                                    (targets_df.index < self.date_range[1])].index[::step]
            targets_temp = targets_df[(targets_df.index >= self.date_range[0]) & \
                                      (targets_df.index < self.date_range[1])].values[::step]
            if (i == 0):
                self.dates = np.array(dates_temp)
                y = targets_temp.astype(np.float32)
            else:
                self.dates = np.concatenate((self.dates, dates_temp), axis=0)
                y = np.concatenate((y, targets_temp), axis=0)
            if i == 0:
                self.dates_len = self.dates.shape[0]
            # For training and testing period, get corresponding dates
            for j, date in enumerate(self.train_start_end_dates):
                dates_temp = targets_df[(targets_df.index >= date[0]) & \
                                        (targets_df.index < date[1])].index[::step]
                if (i == 0) & (j == 0):
                    self.train_dates = np.array(dates_temp)
                else:
                    self.train_dates = np.concatenate((self.train_dates, dates_temp), axis=0)
                if i == 0:
                    self.train_dates_len.append(len(dates_temp))
            for j, date in enumerate(self.test_start_end_dates):
                dates_temp = targets_df[(targets_df.index >= date[0]) & \
                                        (targets_df.index < date[1])].index[::step]
                if (i == 0) & (j == 0):
                    self.test_dates = np.array(dates_temp)
                else:
                    self.test_dates = np.concatenate((self.test_dates, dates_temp), axis=0)
                if i == 0:
                    self.test_dates_len.append(len(dates_temp))

        # Initialize arrays for ensemble model data
        self.alphas = [float(a) for a in self.alphas_eval[1:]]
        self.alphas = [a / 2 for a in self.alphas] + [1 - a / 2 for a in self.alphas]
        self.alphas = sorted(list(set(self.alphas)))
        self.alphas = ['point'] + ['quantile' + str(a) for a in self.alphas]
        model_data = np.zeros(shape=(y.shape[0], len(self.ensemble_models), len(self.alphas)))
        # Get model data
        start_time = timeit.default_timer()
        for i, model in enumerate(self.ensemble_models):
            elapsed = timeit.default_timer() - start_time
            print(f'Start of iteration {i + 1}. Time elapsed {elapsed} seconds.')
            for j, location in enumerate(self.locations):
                for k, date in enumerate(self.dates[j * self.dates_len:(j + 1) * self.dates_len]):
                    for m, alpha in enumerate(self.alphas):
                        date_list = self.f_ensemble[model][location][self.prediction_type[0]][alpha][:]
                        date_list = np.array([a.decode('UTF-8') for a in date_list['target_end_date']])
                        if date in date_list:
                            idx = np.argwhere(date_list == date)
                            if len(idx) > 1:
                                model_data[j * self.dates_len + k, i, m] = \
                                self.f_ensemble[model][location][self.prediction_type[0]][alpha][:]['value'][idx[0]]
                            else:
                                model_data[j * self.dates_len + k, i, m] = \
                                self.f_ensemble[model][location][self.prediction_type[0]][alpha][:]['value'][idx]
                        else:
                            model_data[j * self.dates_len + k, i, m] = np.nan
        return model_data, y

    def _nan_helper(self, y):
        """
        Inputs:
            y: 1D array
        Returns:
            logical indices of NaNs.
            function to return indices of logical indices of NaNs.
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    def get_interp_indices(self, y):
        """
        Inputs:
            :param y: 1D array with nans
        :return:
            Ordered list of lists, where each list contains a contiguous set of indices where y is nan
        """
        nans, x = self._nan_helper(y)
        all_idxs = nans.nonzero()[0]
        split_idxs = []
        split_idxs.append([all_idxs[0]])
        for idx in all_idxs[1:]:
            if split_idxs[-1][-1] == idx - 1:
                split_idxs[-1].append(idx)
            else:
                split_idxs.append([idx])
        return split_idxs

    def fillin_lin_interp_mean_persistence(self, data):
        """
        Fill in NaNs by linearly interpolate along the time axis if the length of the period being
        interpolated is less than self.interp_max. Otherwise, fill in the mean prediction produced by all other models.
        If no models make a prediction, then use the persistence forecast.
        """
        total_size = 0
        #idx_cum = np.concatenate([[0], np.cumsum(self.dates_len)])
        empty_idxs = []
        mean_fill_indices = np.zeros((len(self.ensemble_models), len(self.alphas), len(self.locations)), dtype = object)
        #print(all_nan_indices.shape)
        for i in range(len(self.ensemble_models)):
            for j in range(len(self.alphas)):
                for l in range(len(self.locations)):
                    all_idxs = slice(l * self.dates_len, (l+1) * self.dates_len)
                    y = data[all_idxs, i, j]
                    nans, x = self._nan_helper(y)
                    interp_idxs = self.get_interp_indices(y)
                    non_interp_idxs = []
                    if 0 not in interp_idxs[0] and len(interp_idxs[0]) <= self.interp_max:
                        y[interp_idxs[0]] = np.interp(interp_idxs[0], x(~nans), y[~nans])
                    else:
                        non_interp_idxs.append(0)
                    if self.dates_len-1 not in interp_idxs[-1] and len(interp_idxs[-1]) <= self.interp_max:
                        y[interp_idxs[-1]] = np.interp(interp_idxs[-1], x(~nans), y[~nans])
                    else:
                        non_interp_idxs.append(len(interp_idxs) - 1)
                    for k, idxs in enumerate(interp_idxs[1:-1]):
                        if len(idxs) <= self.interp_max:
                            y[idxs] = np.interp(idxs, x(~nans), y[~nans])
                        else:
                            non_interp_idxs.append(k+1)

                    #print(non_interp_idxs)
                    mean_fill_indices[i,j,l] = [interp_idxs[q] for q in non_interp_idxs]
                    data[all_idxs, i, j] = y
        for i, j, l in product(range(len(self.ensemble_models)), range(len(self.alphas)), range(len(self.locations))):
            all_idxs = slice(l * self.dates_len, (l + 1) * self.dates_len)
            y = data[all_idxs, i, j]
            for idxs in mean_fill_indices[i,j,l]:
                y_sum = np.zeros(y[idxs].shape)
                y_count = np.zeros(y[idxs].shape)
                for k in range(len(self.ensemble_models)):
                    for m, idx in enumerate(idxs):
                        if idx not in list(itertools.chain.from_iterable(mean_fill_indices[k, j, l])):
                            y_sum[m] += data[l * self.dates_len + idx, k, j]
                            y_count[m] += 1
                if np.any(y_count == 0):
                    for m, count in enumerate(y_count):
                        if count == 0:
                            print('No predictions found for alpha %d, location %s, time index %d' % \
                                  (j, self.locations[j], idxs[m]))
                            if idxs[m] < max(mean_fill_indices[i, j, l][0])+1:
                                y[idxs[m]] = y[mean_fill_indices[i,j,l][0][-1]+1]
                            else:
                                y[idxs[m]] = y[idxs[m]-1]
                        else:
                            y[idxs[m]] = y_sum[m] / count
                else:
                    y[idxs] = y_sum / y_count
            data[all_idxs, i, j] = y
        return data

    def rescale_inv_max(self, y):
        """
        Rescale data for each location by dividing by the max value for that location in the
        true (training) data set. If the location is not in the training set, use the true value from the test set.
        """
        self.rescale_factors = {}
        for l, loc in enumerate(self.locations):
            if loc in self.train_locations:
                dates_ind_dict = dict((k, i + l * self.dates_len) for i, k in \
                                      enumerate(self.dates[l * self.dates_len:(l + 1) * self.dates_len]))
                train_inter = set(dates_ind_dict).intersection( \
                    self.train_dates[l * sum(self.train_dates_len):(l + 1) * sum(self.train_dates_len)])
                train_indices = np.sort(np.array([dates_ind_dict[x] for x in train_inter]))

                #idx = slice(l * sum(self.dates_len), (l + 1) * sum(self.dates_len))
                norm = np.amax(y[train_indices, :])
                # y[idx, :] /= norm
                # self.model_data[idx, :, :] /= norm
                self.rescale_factors[loc] = norm
            else:
                dates_ind_dict = dict((k, i + l * self.dates_len) for i, k in \
                                      enumerate(self.dates[l * self.dates_len:(l + 1) * self.dates_len]))
                test_inter = set(dates_ind_dict).intersection( \
                    self.test_dates[l * sum(self.test_dates_len):(l + 1) * sum(self.test_dates_len)])
                test_indices = np.sort(np.array([dates_ind_dict[x] for x in test_inter]))

                # idx = slice(l * sum(self.dates_len), (l + 1) * sum(self.dates_len))
                norm = np.amax(y[test_indices, :])
                # self.y[idx, :] /= norm
                # self.model_data[idx, :, :] /= norm
                self.rescale_factors[loc] = norm


    def train_test_split(self, model_data, y):
        """
        Inputs:
            :param model_data:
            :param y:
        :return:

        Splits the model data and y into their respective portions, and saves them to the class.
        """
        locations_ind_dict = dict((k,i) for i,k in enumerate(self.locations))
        train_l_inter = set(locations_ind_dict).intersection(self.train_locations)
        train_l_indices = np.sort(np.array([locations_ind_dict[x] for x in train_l_inter]))
        test_l_inter = set(locations_ind_dict).intersection(self.test_locations)
        test_l_indices = np.sort(np.array([locations_ind_dict[x] for x in test_l_inter]))
        for l in train_l_indices:
            dates_ind_dict = dict((k,i+l*self.dates_len) for i,k in \
                                  enumerate(self.dates[l*self.dates_len:(l+1)*self.dates_len]))

            train_inter = set(dates_ind_dict).intersection(\
                self.train_dates[l*sum(self.train_dates_len):(l+1)*sum(self.train_dates_len)])
            train_indices = np.sort(np.array([dates_ind_dict[x] for x in train_inter]))

            if l == 0:
                self.train_model_data = model_data[train_indices]
                self.train_y          = y[train_indices]
            else:
                self.train_model_data = np.concatenate((self.train_model_data,
                                                   model_data[train_indices]), axis = 0)
                self.train_y          = np.concatenate((self.train_y,
                                                   y[train_indices]), axis = 0)
        for l in test_l_indices:
            dates_ind_dict = dict((k, i + l * self.dates_len) for i, k in \
                                  enumerate(self.dates[l * self.dates_len:(l + 1) * self.dates_len]))

            test_inter = set(dates_ind_dict).intersection( \
                self.test_dates[l * sum(self.test_dates_len):(l + 1) * sum(self.test_dates_len)])
            test_indices = np.sort(np.array([dates_ind_dict[x] for x in test_inter]))

            if l == 0:
                self.test_model_data = model_data[test_indices]
                self.test_y = y[test_indices]
            else:
                self.test_model_data = np.concatenate((self.test_model_data,
                                                   model_data[test_indices]), axis=0)
                self.test_y = np.concatenate((self.test_y,
                                          y[test_indices]), axis=0)







class CovidDataLoader(d2l.DataModule):
    def __init__(self, covid_data, time_delays, batch_size=32, val_size=32, dtype=torch.float32):
        super().__init__()
        self.save_hyperparameters(ignore=['covid_data'])
        self.alphas_eval = covid_data.alphas_eval
        rescale = False
        try:
            self.rescale_factors = covid_data.rescale_factors
            rescale = True
        except:
            print('Training and testing data have not been rescaled.')
        if rescale:
            for k, loc in enumerate(covid_data.train_locations):
                idx = slice(k * sum(covid_data.train_dates_len), (k + 1) * sum(covid_data.train_dates_len))
                covid_data.train_y[idx, :] /= self.rescale_factors[loc]
                covid_data.train_model_data[idx, :] /= self.rescale_factors[loc]
            for k, loc in enumerate(covid_data.test_locations):
                idx = slice(k * sum(covid_data.test_dates_len), (k + 1) * sum(covid_data.test_dates_len))
                covid_data.test_y[idx, :] /= self.rescale_factors[loc]
                covid_data.test_model_data[idx, :] /= self.rescale_factors[loc]

        self.dates_train, self.dates_test = [], []
        idx = np.concatenate([[0], np.cumsum(covid_data.train_dates_len)])
        for i, _ in enumerate(covid_data.train_dates_len):
            self.dates_train.append(covid_data.train_dates[(idx[i] + time_delays):idx[i + 1]])
        self.dates_train = np.concatenate(self.dates_train)
        idx = np.concatenate([[0], np.cumsum(covid_data.test_dates_len)])
        for i, _ in enumerate(covid_data.test_dates_len):
            self.dates_test.append(covid_data.test_dates[(idx[i] + time_delays):idx[i + 1]])
        self.dates_test = np.concatenate(self.dates_test)
        self.n_loc_train = len(covid_data.train_locations)
        self.n_loc_test = len(covid_data.test_locations)
        # Create queries, keys, values for training data, covid_train
        self.n_train = (len(self.dates_train) - len(covid_data.train_dates_len)) * self.n_loc_train + 1
        y_train = covid_data.train_y
        self.queries_train = torch.zeros((self.n_train,
                                          1,
                                          y_train.shape[1] * self.time_delays))
        self.keys_train = torch.zeros((self.n_train,
                                       covid_data.train_model_data.shape[1],
                                       covid_data.train_model_data.shape[2] * self.time_delays))
        self.values_train = torch.zeros((self.n_train,
                                         covid_data.train_model_data.shape[1],
                                         covid_data.train_model_data.shape[2]))
        self.y_train = torch.zeros((self.n_train, 1, 1))
        idx_cum = np.concatenate([[0], np.cumsum([a - 1 - self.time_delays for a in covid_data.train_dates_len])])
        idx_cum_2 = np.concatenate([[0], np.cumsum(covid_data.train_dates_len)])
        for i, _ in enumerate(covid_data.train_locations):
            for j in range(len(covid_data.train_dates_len)):
                for delay in range(self.time_delays):
                    idx = slice(i * (len(self.dates_train) - (len(idx_cum) - 1)) + idx_cum[j], 
                                i * (len(self.dates_train) - (len(idx_cum) - 1)) + idx_cum[j + 1])
                    try:
                        self.queries_train[idx, 0, delay * y_train.shape[1]:(delay + 1) * y_train.shape[1]] = \
                            torch.from_numpy(y_train[
                                         (i * sum(covid_data.train_dates_len) + idx_cum_2[j] + self.time_delays - delay):(
                                                 i * sum(covid_data.train_dates_len) + idx_cum_2[j + 1] - delay - 1), :])
                    except:
                        print('Exception for i = %d, j = %d' % (i, j))
                    self.keys_train[idx, :,
                    delay * covid_data.train_model_data.shape[2]:(delay + 1) * covid_data.train_model_data.shape[2]] = \
                        torch.from_numpy(covid_data.train_model_data[
                                         (i * sum(covid_data.train_dates_len) + idx_cum_2[j] + self.time_delays - delay):(
                                                 i * sum(covid_data.train_dates_len) + idx_cum_2[j + 1] - delay - 1), :,
                                         :])
                    self.keys_train[idx, :, delay * covid_data.train_model_data.shape[2]] -= \
                        torch.from_numpy(y_train[
                                         (i * sum(covid_data.train_dates_len) + idx_cum_2[
                                             j] + self.time_delays - delay - 1):(
                                                 i * sum(covid_data.train_dates_len) + idx_cum_2[j + 1] - delay - 2), :])
                    self.values_train[idx, :, :] = torch.from_numpy(
                        covid_data.train_model_data[(i * sum(covid_data.train_dates_len) + self.time_delays + idx_cum_2[j] + 1):(
                                i * sum(covid_data.train_dates_len) + idx_cum_2[j + 1]), :, :])
                    self.y_train[idx, 0, 0] = torch.squeeze(torch.from_numpy(
                        y_train[(i * sum(covid_data.train_dates_len) + self.time_delays + idx_cum_2[j] + 1):(
                                i * sum(covid_data.train_dates_len) + idx_cum_2[j + 1])]))
        self.queries_train = self.queries_train.type(dtype)
        self.keys_train = self.keys_train.type(dtype)
        self.values_train = self.values_train.type(dtype)
        # Create queries, keys, values for test data, covid_test
        self.n_test = (len(self.dates_test) - len(covid_data.test_dates_len)) * self.n_loc_test
        y_test = covid_data.test_y
        self.queries_test = torch.zeros((self.n_test,
                                         1,
                                         y_test.shape[1] * self.time_delays))
        self.keys_test = torch.zeros((self.n_test,
                                      covid_data.test_model_data.shape[1],
                                      covid_data.test_model_data.shape[2] * self.time_delays))
        self.values_test = torch.zeros((self.n_test,
                                        covid_data.test_model_data.shape[1],
                                        covid_data.test_model_data.shape[2]))
        self.y_test = torch.zeros((self.n_test, 1, 1))
        idx_cum = np.concatenate([[0], np.cumsum([a - 1 - time_delays for a in covid_data.test_dates_len])])
        idx_cum_2 = np.concatenate([[0], np.cumsum(covid_data.test_dates_len)])
        for i, _ in enumerate(covid_data.test_locations):
            for j in range(len(covid_data.test_dates_len)):
                for delay in range(self.time_delays):
                    idx = slice(i * (len(self.dates_test) - (len(idx_cum) - 1)) + idx_cum[j],
                                i * (len(self.dates_test) - (len(idx_cum) - 1)) + idx_cum[j + 1])
                    self.queries_test[idx, 0, delay * y_test.shape[1]:(delay + 1) * y_test.shape[1]] = \
                        torch.from_numpy(y_test[
                                         (i * sum(covid_data.test_dates_len) + idx_cum_2[j] + self.time_delays - delay):(
                                                 i * sum(covid_data.test_dates_len) + idx_cum_2[j + 1] - delay - 1), :])
                    self.keys_test[idx, :,
                    delay * covid_data.test_model_data.shape[2]:(delay + 1) * covid_data.test_model_data.shape[2]] = \
                        torch.from_numpy(covid_data.test_model_data[
                                         (i * sum(covid_data.test_dates_len) + idx_cum_2[j] + self.time_delays - delay):(
                                                 i * sum(covid_data.test_dates_len) + idx_cum_2[j + 1] - delay - 1), :,
                                         :])
                    self.keys_test[idx, :, delay * covid_data.test_model_data.shape[2]] -= \
                        torch.from_numpy(y_test[
                                         (i * sum(covid_data.test_dates_len) + idx_cum_2[j] + self.time_delays - delay - 1):(
                                                 i * sum(covid_data.test_dates_len) + idx_cum_2[j + 1] - delay - 2), :])
                    self.values_test[idx, :, :] = torch.from_numpy(
                        covid_data.test_model_data[(i * sum(covid_data.test_dates_len) + self.time_delays + idx_cum_2[j] + 1):(
                                i * sum(covid_data.test_dates_len) + idx_cum_2[j + 1]), :, :])
                    self.y_test[idx, 0, 0] = torch.squeeze(
                        torch.from_numpy(y_test[(i * sum(covid_data.test_dates_len) + self.time_delays + idx_cum_2[j] + 1):(
                                i * sum(covid_data.test_dates_len) + idx_cum_2[j + 1])]))
        self.queries_test = self.queries_test.type(dtype)
        self.keys_test = self.keys_test.type(dtype)
        self.values_test = self.values_test.type(dtype)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        if train:
            return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
        else:
            return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train,
                                               sampler=torch.utils.data.SequentialSampler(dataset))

    def get_dataloader(self, train, shuffle_train=True):
        i = slice(0, self.n_train) if train else slice(0, self.n_test)
        if train:
            return self.get_tensorloader((self.queries_train, self.keys_train, self.values_train, self.y_train),
                                         shuffle_train, i)
        else:
            return self.get_tensorloader((self.queries_test, self.keys_test, self.values_test, self.y_test), train, i)


class TeacherForcingData(LorenzDataModule):
    def __init__(self, data_in, trained_model, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0, val_size = 64):
        super().__init__()
        self.num_train = num_train
        data = trained_model.forward(data_in.queries, data_in.keys, data_in.values)
        self.times = data_in.times[1:]
        self.model_zoo = data_in.model_zoo
        model_data = torch.zeros(data.size(0)-1, len(self.model_zoo), data.size(2))
        for j, model in enumerate(self.model_zoo):
            model_data[:,j] = model.run_array(data[:-1].squeeze(1))
        self.values  = model_data[1:]
        self.queries = data[1:-1]
        self.keys    = model_data[:-1] - self.queries
        self.y       = data_in.y[2:]
        self.save_hyperparameters()

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)

class TimeSeriesAttention(d2l.Module):
    def __init__(self):
        super().__init__()
        self.board = ProgressBoardVT(ylim = [None, [-20,20]], legend_loc = ['upper right', 'upper right'])

    def validation_step_noplot(self, batch, model_zoo):
        l, pred = self.validation_loss(self.predict(*batch[:-1], model_zoo), batch[-1])
        return l, pred

    def validation_loss(self, y_hat, y, cutoff = 40.):
        err = torch.mean((y_hat.squeeze(1) - y.squeeze(1))**2, 1)
        vt_arr  = torch.arange(0, len(err))[err > cutoff]
        if len(vt_arr) == 0:
            return len(err), y_hat
        else:
            return max(vt_arr[0] - 1, 0), y_hat

    def predict(self, queries_val, keys_val, values_val, model_zoo):
        y_hat    = torch.zeros(queries_val.size(0),
                               queries_val.size(1),
                               values_val.size(2))
        #print(y_hat.size())
        num_features = values_val.size(-1)
        feature_size = keys_val.size(-1)
        queries_next, keys_next, values_next = queries_val[0].unsqueeze(0),\
                                               keys_val[0].unsqueeze(0),\
                                               values_val[0].unsqueeze(0)
        y_hat[0] = self.forward(queries_next, keys_next, values_next)
        if feature_size > num_features:
            for k in range(queries_val.size(0)-1):
                keys_next[:,:,num_features:]    = keys_next[:,:,:-num_features].clone()
                queries_next[:,:,num_features:] = queries_next[:,:,:-num_features].clone()
                keys_next[:,:,:num_features]    = values_next - y_hat[k].unsqueeze(0)
                queries_next[:,:,:num_features] = y_hat[k].unsqueeze(0)
                #print(queries_next.size())
                #print(queries_next.reshape(-1).size())
                for j, model in enumerate(model_zoo):
                    values_next[:,j] = model.forward(queries_next[:,:,:num_features].reshape(-1))
                y_hat[k+1] = self.forward(queries_next,keys_next,values_next)
        else:
            for k in range(queries_val.size(0)-1):
                keys_next = values_next - y_hat[k].unsqueeze(0)
                queries_next = y_hat[k].unsqueeze(0)
                # print(queries_next.size())
                # print(queries_next.reshape(-1).size())
                for j, model in enumerate(model_zoo):
                    values_next[:, j] = model.forward(queries_next.reshape(-1))
                y_hat[k + 1] = self.forward(queries_next, keys_next, values_next)

        return y_hat


    def plot(self, key, value, train, label_val = None):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = ['epoch', 'pred step']
        self.board.ylabel = ['MSE', 'x(t)']
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
            self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                            ('train_' if train else 'val_') + key, train = train,
                            every_n=int(n), label_val = label_val)
        else:
            x = np.arange(value.size(0))
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
            self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                            ('train_' if train else 'val_') + key, train = train,
                            every_n=int(n), label_val = label_val)

class AdditiveAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, num_hiddens, dropout, lr, l1_reg = 0.001, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(feature_size, num_hiddens, bias = True)
        self.W_q = torch.nn.Linear(feature_size, num_hiddens, bias = False)
        self.w_v = torch.nn.Linear(num_hiddens, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr
        self.l1_reg = l1_reg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries, W_keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = self.w_v(features)
        self.attention_weights = self.sm(scores.squeeze(-1))
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2 + \
            self.l1_reg * sum(torch.linalg.norm(p, 1) for p in self.parameters())
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class CovidAdditiveAttention(d2l.Module):
    """Additive attention."""
    def __init__(self, query_size, key_size, num_hiddens, dropout, lr, alphas, l1_reg = 0.001, **kwargs):
        super(CovidAdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias = True)
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias = False)
        self.w_v = torch.nn.Linear(num_hiddens, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr
        self.l1_reg = l1_reg
        self.alphas = alphas
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries, W_keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = self.w_v(features)
        self.attention_weights = self.sm(scores.squeeze(-1))
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def loss(self, y_hat, y):
        return weighted_interval_score(y_hat, y, self.alphas).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class CovidMultiValueAdditiveAttention(d2l.Module):
    """Multi-Value Additive attention."""
    def __init__(self, query_size, key_size, value_size, num_hiddens, dropout, lr, alphas, l1_reg = 0.001, **kwargs):
        super(CovidMultiValueAdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(key_size, num_hiddens * value_size, bias = True)
        self.W_q = torch.nn.Linear(query_size, num_hiddens * value_size, bias = False)
        self.w_v = torch.nn.Linear(num_hiddens, value_size, bias = False)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -2)
        self.lr = lr
        self.l1_reg = l1_reg
        self.alphas = alphas
        self.value_size = value_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries, W_keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed).reshape(
            features_unsqueezed.size(0), features_unsqueezed.size(1),
            features_unsqueezed.size(2), self.value_size, -1
        )
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = torch.sum(features * self.w_v.weight.reshape(1, 1, 1, self.value_size, -1), dim = -1)
        self.attention_weights = self.sm(scores)
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.sum(self.dropout(self.attention_weights) * values.unsqueeze(1), dim = -2, keepdim = False)

    def loss(self, y_hat, y):
        return weighted_interval_score(y_hat, y, self.alphas).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class CovidMultiHeadAdditiveAttention(d2l.Module):
    """Multi-Value Additive attention."""
    def __init__(self, query_size, key_size, value_size, num_heads, num_hiddens,
                 dropout, lr, alphas, l1_reg = 0.001, **kwargs):
        super(CovidMultiHeadAdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(key_size, num_hiddens * num_heads, bias = True)
        self.W_q = torch.nn.Linear(query_size, num_hiddens * num_heads, bias = False)
        self.w_v = torch.nn.Linear(num_hiddens, num_heads, bias = False)
        self.W_out = torch.nn.Linear(num_heads * value_size, value_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = 2)
        self.lr = lr
        self.l1_reg = l1_reg
        self.alphas = alphas
        self.value_size = value_size
        self.num_heads  = num_heads
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries, W_keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed).reshape(
            features_unsqueezed.size(0), features_unsqueezed.size(1),
            features_unsqueezed.size(2), self.num_heads, -1
        )
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = torch.sum(features * self.w_v.weight.reshape(1, 1, 1, self.num_heads, -1), dim = -1)
        self.attention_weights = self.sm(scores).squeeze(1).transpose(1,2)
        concat_out = torch.bmm(self.dropout(self.attention_weights), values).reshape(values.size(0), 1, -1)
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return self.W_out(concat_out)

    def loss(self, y_hat, y):
        return weighted_interval_score(y_hat, y, self.alphas).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class NoAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, num_hiddens, output_size, dropout, lr, l1_reg = 0.001, **kwargs):
        super(NoAttention, self).__init__(**kwargs)
        #self.W_k = torch.nn.Linear(feature_size, num_hiddens, bias = True)
        self.W_q = torch.nn.Linear(feature_size, num_hiddens, bias = True)
        self.w_v = torch.nn.Linear(num_hiddens, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        #self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr
        self.l1_reg = l1_reg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries = self.W_q(queries)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        #features_unsqueezed = W_queries.unsqueeze(2)
        features = torch.tanh(W_queries)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = self.w_v(features)
        #self.attention_weights = self.sm(scores.squeeze(-1))
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return scores

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2 + \
            self.l1_reg * sum(torch.linalg.norm(p, 1) for p in self.parameters())
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class InitAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, num_hiddens, dropout, lr, l1_reg = 0.001, **kwargs):
        super(InitAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(feature_size, num_hiddens, bias = True)
        self.W_q = torch.nn.Linear(feature_size, num_hiddens, bias = False)
        self.w_v = torch.nn.Linear(num_hiddens, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr
        self.l1_reg = l1_reg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, queries, keys, values):
        W_queries, W_keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        #self.attention_weights = torch.matmul(features, self.w_v).squeeze(-1)
        scores = self.w_v(features)
        self.attention_weights = self.sm(scores.squeeze(-1))

        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2 + \
            self.l1_reg * sum(torch.linalg.norm(p, 1) for p in self.parameters())
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

class IdiotAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, output_size, dropout, lr, **kwargs):
        super(IdiotAttention, self).__init__(**kwargs)
        self.W   = torch.nn.Linear(feature_size, output_size, bias = True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lr = lr

    def forward(self, queries, keys, values):
        self.feature = torch.cat((values, keys[:,:,:-values.shape(2)]),
                                 dim = 2).reshape(values.shape(0), -1)
        return self.W(self.feature).unsqueeze(1)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def exact_solve(self, queries, keys, values, y):
        with torch.no_grad():
            _ = self.forward(queries, keys, values)
            bias_feature = torch.cat((self.feature, torch.ones(self.feature.size(0),1)),1)
            info_mat     = torch.mm(bias_feature.transpose(0,1), bias_feature).detach().numpy()
            target_mat   = torch.mm(bias_feature.transpose(0,1), y.squeeze(1)).detach().numpy()
            print(info_mat.shape)
            print(target_mat.shape)
            Wout         = np.linalg.solve(info_mat, target_mat)
            self.W.weight= torch.nn.Parameter(torch.from_numpy(Wout[:-1].T))
            self.W.bias  = torch.nn.Parameter(torch.from_numpy(Wout[-1]))

class CovidIdiotAttention(d2l.Module):
    """Additive attention."""
    def __init__(self, feature_size, output_size, dropout, lr, alphas, l1_reg = 0., **kwargs):
        super(CovidIdiotAttention, self).__init__(**kwargs)
        self.W   = torch.nn.Linear(feature_size, output_size, bias = True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lr = lr
        self.alphas = alphas
        self.l1_reg = l1_reg

    def forward(self, queries, keys, values):
        self.feature = torch.cat((values, keys[:,:,:-values.size(2)]),
                                 dim = 2).reshape(values.size(0), -1)
        return self.W(self.feature).unsqueeze(1)

    def loss(self, y_hat, y):
        return weighted_interval_score(y_hat, y, self.alphas).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def exact_solve(self, queries, keys, values, y):
        with torch.no_grad():
            _ = self.forward(queries, keys, values)
            bias_feature = torch.cat((self.feature, torch.ones(self.feature.size(0),1)),1)
            info_mat     = torch.mm(bias_feature.transpose(0,1), bias_feature).detach().numpy()
            target_mat   = torch.mm(bias_feature.transpose(0,1), y.squeeze(1)).detach().numpy()
            print(info_mat.shape)
            print(target_mat.shape)
            Wout         = np.linalg.solve(info_mat, target_mat)
            self.W.weight= torch.nn.Parameter(torch.from_numpy(Wout[:-1].T))
            self.W.bias  = torch.nn.Parameter(torch.from_numpy(Wout[-1]))

class LLRAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, output_size, **kwargs):
        super(LLRAttention, self).__init__(**kwargs)
        self.W   = torch.nn.Linear(feature_size, 1, bias = False)
        self.output_size = output_size

    def forward(self, queries, keys, values):
        scale_mat = torch.mm(torch.ones(values.size(0),1), self.W.weight)
        #print(scale_mat.unsqueeze(1).size())
        #print(values.size())
        return torch.bmm(scale_mat.unsqueeze(1), values)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 0.)

    def predict(self, queries_val, keys_val, values_val, model_zoo):
        y_hat = torch.zeros(queries_val.size(0),
                            queries_val.size(1),
                            values_val.size(2))
        self.get_weights(queries_val, keys_val)
        queries_next, keys_next, values_next = queries_val[0].unsqueeze(0), \
                                               keys_val[0].unsqueeze(0), \
                                               values_val[0].unsqueeze(0)
        y_hat[0] = self.forward(queries_next, keys_next, values_next)
        for k in range(queries_val.size(0) - 1):
            keys_next = values_next
            queries_next = y_hat[k].unsqueeze(0)
            # print(queries_next.size())
            # print(queries_next.reshape(-1).size())
            for j, model in enumerate(model_zoo):
                values_next[:, j] = model.forward(queries_next.reshape(-1))
            y_hat[k + 1] = self.forward(queries_next, keys_next, values_next)

        return y_hat

    def get_weights(self, queries, keys):
        with torch.no_grad():
            info_mat     = np.double(torch.mm(keys[0], keys[0].transpose(0,1)).detach().numpy())
            target_mat   = np.double(torch.mm(queries[0], keys[0].transpose(0,1)).detach().numpy())
            #print(self.W.weight)
            #print(info_mat.shape)
            #print(target_mat.shape)
            prob_constr = LinearConstraint(np.ones(info_mat.shape[0]), 1., 1.)
            hessp = lambda x, p: np.zeros(x.size)
            min_fun = lambda x: np.mean((x @ info_mat - target_mat)**2.0)
            Wout         = minimize(min_fun, np.ones((1, info_mat.shape[0]))/info_mat.shape[0],
                                      method = 'trust-constr', hessp = hessp,
                                      bounds = [(0,1)]*info_mat.shape[0],
                                      constraints = prob_constr)
            #print(Wout.x)
            #Wout         = target_mat @ np.linalg.pinv(info_mat)
            self.W.weight= torch.nn.Parameter(torch.from_numpy(Wout.x.reshape(1,-1)).type(torch.float32))
            #print(self.W.weight)

class DotProductAttention(TimeSeriesAttention):
    """Scaled dot product attention."""
    def __init__(self, feature_size, dropout, lr, num_hidden=None, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.lr = lr
        self.sm = torch.nn.Softmax(dim=-1)
        self.num_hidden = num_hidden
        if self.num_hidden is not None:
            self.W_q = torch.nn.Linear(feature_size, num_hidden, bias=True)
            self.W_k = torch.nn.Linear(feature_size, num_hidden, bias=True)

    # Shape of queries: (batch_size, no. queries, feature_size)
    # Shape of keys: (batch_size, no. key-value pairs, feature_size)
    # Shape of values: (batch_size, no. key-value pairs, feature_size)
    def forward(self, queries, keys, values):
        # Swap last two dimensions of keys
        d = queries.shape[-1]
        if self.num_hidden:
            queries, keys = self.W_q(queries), self.W_k(keys)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)
        self.attention_weights = self.sm(scores)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1))**2 / 2
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class TestAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, num_hiddens, dropout, lr, sigma=0.01, **kwargs):
        super(TestAttention, self).__init__(**kwargs)
        self.W_k = torch.normal(0, sigma, (feature_size, num_hiddens), requires_grad=True)
        self.W_q = torch.normal(0, sigma, (feature_size, num_hiddens), requires_grad=True)
        self.w_v = torch.normal(0, sigma, (num_hiddens, 1), requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr

    def forward(self, queries, keys, values):
        W_queries, W_keys = torch.matmul(queries, self.W_q), torch.matmul(keys, self.W_k)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features_unsqueezed = W_queries.unsqueeze(2) + W_keys.unsqueeze(1)
        features = torch.tanh(features_unsqueezed)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = torch.matmul(features, self.w_v)
        self.attention_weights = self.sm(scores.squeeze(-1))
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return values[:,4].unsqueeze(1)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2
        return l.mean()

    def configure_optimizers(self):
        return torch.optim.Adam([self.W_k, self.W_q, self.w_v], self.lr)

class TrainerAttentionVT(d2l.Trainer):

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim[0] = [0, self.max_epochs]
        model.board.xlim[1] = [0,self.val_size]
        self.model = model.to(model.device)

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return tuple([batch_elem.to(self.model.device) for batch_elem in batch])

    def prepare_batch_val(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch
    def fit(self, model, data):
        self.val_size = data.val_size
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch(data.model_zoo)
    def fit_epoch(self, model_zoo):
            #tic = time.perf_counter()
            self.fit_epoch_base()
            #toc = time.perf_counter()
            #print('Iter runtime: %0.3e sec.' % (toc - tic))
            if self.epoch % 25 == 0:
                self.mean_validation_loss(model_zoo)
    def fit_epoch_base(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward(retain_graph=True)
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()

    def mean_validation_loss(self, model_zoo):
        loss_all = []
        self.model = self.model.to("cpu")
        for k, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                if k == 0:
                    l, pred = self.model.validation_step_noplot(self.prepare_batch_val(batch), model_zoo)
                else:
                    l, tmp = self.model.validation_step_noplot(self.prepare_batch_val(batch), model_zoo)
                loss_all.append(l)
                self.val_batch_idx = k+1
            if self.epoch == 0 and k == 0:
                self.model.plot('true', self.prepare_batch_val(batch)[-1][:,0,0], train = False)
        vt, confidence = median_confidence(loss_all)
        self.model.plot('median_vt', pred[:,0,0], train=False, label_val = (vt, confidence))
        self.model = self.model.to(self.model.device)
        # self.model.plot('vt', (loss_sum / self.val_batch_idx), train=False)

class CovidTrainer(d2l.Trainer):
    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return tuple([batch_elem.to(self.model.device) for batch_elem in batch])
        #return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        #tic = time.perf_counter()
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
        #toc = time.perf_counter()
        #print('Epoch Runtime: %s sec.' % (toc - tic))

    def prepare_data(self, data):
        self.train_dataloader  = data.train_dataloader()
        self.val_dataloader    = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches   = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)




