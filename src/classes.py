import numpy as np
from numba import int32, float64
from numba.experimental import jitclass
from d2l import torch as d2l
import torch

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
        model_output[0] = self.state
        for i in range(T + discard_len):
            model_output[i + 1] = self.forward()

        return model_output[discard_len:]

    def forward(self):
        for i in range(self.int_steps):
            self.state, self.time = self.rk4()
        return self.state

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

class LorenzDataModule(d2l.DataModule):
    def __init__(self, val_size = 64):
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


class LorenzPeriodicRhoData(LorenzDataModule):
    def __init__(self, true_model, model_zoo, noise = 0., num_train = 1000, num_val = 1000, num_discard = 100,
                 batch_size = 32, tau=0.1, int_steps=10, time=0., period=100., sigma=10.,
                 beta=8 / 3, ic=np.array([]), ic_seed=0):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val + num_discard + 1
        self.data_model = true_model
        self.model_zoo = model_zoo
        data = self.data_model.run(n, num_discard)
        model_data = np.zeros((data.shape[0]-1, len(model_zoo), data.shape[1]))
        for j, model in enumerate(model_zoo):
            model_data[:,j] = model.run_array(data[:-1])
        self.keys    = torch.from_numpy(model_data[:-1])
        self.values  = torch.from_numpy(model_data[1:])
        self.queries = torch.from_numpy(data[1:-1]).unsqueeze(1)
        self.y       = torch.from_numpy(data[2:]).unsqueeze(1)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.queries, self.keys, self.values, self.y), train, i)

class AdditiveAttention(d2l.Module):
    """Additive attention."""
    def __init__(self, feature_size, num_hiddens, dropout, lr, sigma=0.01, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.normal(0, sigma, (feature_size, num_hiddens), requires_grad=True)
        self.W_q = torch.normal(0, sigma, (feature_size, num_hiddens), requires_grad=True)
        self.w_v = torch.normal(0, sigma, (num_hiddens, 1), requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim = -1)
        self.lr = lr
        self.board = d2l.ProgressBoard(yscale = 'log')

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
        return torch.bmm(self.dropout(self.attention_weights), values)

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) **2 /2
        return l.mean()

    def validation_step(self, batch):
        l = self.validation_loss(self.predict(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def validation_loss(self, y_hat, y, cutoff = 1.0):
        err = torch.mean((y_hat.squeeze(1) - y.squeeze(1))**2, 1)
        vt_arr  = torch.arange(0, len(err))[err > cutoff]
        if len(vt_arr) == 0:
            return len(err)
        else:
            return max(vt_arr[0] - 1, 0)

    def configure_optimizers(self):
        return torch.optim.Adam([self.W_k, self.W_q, self.w_v], self.lr)

