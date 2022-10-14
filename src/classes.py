import numpy as np
from numba import int32, float64
from numba.experimental import jitclass
from d2l import torch as d2l
import torch
from src.helpers import median_confidence
import time
from statistics import median

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
        self.times = times
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

class IdiotAttention(TimeSeriesAttention):
    """Additive attention."""
    def __init__(self, feature_size, output_size, dropout, lr, **kwargs):
        super(IdiotAttention, self).__init__(**kwargs)
        self.W   = torch.nn.Linear(feature_size, output_size, bias = True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lr = lr

    def forward(self, queries, keys, values):
        self.feature = torch.cat((queries.squeeze(1),
                             (keys - queries).reshape(keys.size(0),-1),
                             values.reshape(values.size(0),-1)), 1)
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
            #tic = time.perf_counter()
            self.fit_epoch()
            #toc = time.perf_counter()
            #print('Iter runtime: %0.3e sec.' % (toc - tic))
            if self.epoch % 25 == 0:
                self.mean_validation_loss(data.model_zoo)
    def fit_epoch(self):
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