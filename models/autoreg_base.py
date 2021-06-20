import torch
import utils

class AutoregressiveBase(torch.nn.Module):
    """Autoregressive bernoulli sampler based on LSTM"""
    
    def __init__(self, n_nodes, hidden_dim=48, n_layers=3, temp_rsample=0.1, device = 'cpu'):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim_out = n_nodes*(n_nodes-1)
        self.hidden_dim = hidden_dim
        self.n_classes = 1
        self.n_layers = n_layers
        self.temp_rsample = temp_rsample
        self.device = device

        self.rnn = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=n_layers, batch_first=True)
        self.proj = torch.nn.Linear(self.hidden_dim, self.n_classes)
        self.embed = torch.nn.Linear(self.n_classes, self.hidden_dim)

        self.h0 = torch.nn.Parameter(1e-3*torch.randn(1, self.n_layers, self.hidden_dim))
        self.c0 = torch.nn.Parameter(1e-3*torch.randn(1, self.n_layers, self.hidden_dim))

        # create variable for the initial input of the LSTM
        self._init_input_param = torch.nn.Parameter(torch.zeros(1, 1, self.n_classes))

    def forward(self, inputs, state):
        inputs = self.embed(inputs)
        out, state = self.rnn(inputs, self._t(state))
        state = self._t(state)
        logit = self.proj(out)
        return logit, state

    def sample(self, size, return_states=False, start_step=0, start_state = None, init_input = None):
        samples, states, logits = self._sample(size, reparametrized=False, start_step=start_step, start_state=start_state, init_input = init_input)
        if return_states:
            return samples, states, logits
        else:
            return samples

    def rsample(self, size, return_states=False, temp=None, hard=False, start_step=0, start_state=None, init_input = None):
        samples, states, logits = self._sample(size, reparametrized=True, temp=temp, hard=hard, start_step=start_step, start_state=start_state, init_input = init_input)
        if return_states:
            return samples, states, logits
        else:
            return samples

    def _sample(self, batch_size, reparametrized=False, temp=None, hard=False, start_step=0, start_state=None, init_input = None):
        assert len(batch_size) == 1
        batch_size = batch_size[0]

        if temp is None:
            temp = [self.temp_rsample]*self.n_dim_out
        if start_state is None:
            state = self._get_state(batch_size) # hidden / cell state at t=0
        else:
            state = start_state
        if init_input is None:
            input = self._init_input(batch_size) # input at t=0
        else:
            input = init_input

        sampled_tokens = []
        state_array_1 = []
        state_array_2 = []
        logit_array = []

        for t in range(start_step, self.n_dim_out):
            logits, state = self.forward(input, state)
            if reparametrized:
                _sample = torch.distributions.RelaxedBernoulli(temperature=temp[t], logits=logits).rsample()
            else:
                _sample = torch.distributions.Bernoulli(logits=logits).sample()
            input = _sample
            sampled_tokens.append(_sample)
            state_array_1.append(state[0])
            state_array_2.append(state[1])
            logit_array.append(logits)
        
        samples = torch.cat(sampled_tokens, dim=1)
        states = [torch.stack(state_array_1, dim=1), torch.stack(state_array_2, dim=1)]
        logits =  logit_array
        if reparametrized and hard:
            samples_hard = utils.one_hot(torch.argmax(samples, -1), 2)
            samples = (samples_hard - samples).detach() + samples
        
        return samples, states, logits


    def log_prob(self, value, return_logits = False):
        batch_size, n_dim_out_value,_ = value.shape
        assert n_dim_out_value == self.n_dim_out
        # add start value
        state = self._get_state(batch_size) # hidden / cell state at t=0
        input = self._init_input(batch_size) # input at t=0
        value = torch.cat([input, value], dim=-2)
        logits, _ = self.forward(value, state)
        logits = logits[:, :-1, :]
        value = value[:, 1:]
        log_probs = torch.distributions.Bernoulli(logits = logits).log_prob(value).sum(1)
        if return_logits:
            return log_probs, logits
        return log_probs

    def entropy(self, n_samples=10 ** 6):
        bs = 100000
        curr = 0
        ent = 0.
        while curr < n_samples:
            curr_batch_size = min(bs, n_samples-curr)
            ent -= torch.sum(self.log_prob(self.sample([curr_batch_size])))
            curr += curr_batch_size
        return ent/n_samples

    def _get_state(self, batch_size=1):
        return (self.h0.repeat(batch_size, 1, 1), self.c0.repeat(batch_size, 1, 1))

    def _init_input(self, batch_size):
        return self._init_input_param.expand(batch_size, 1, self.n_classes)

    def mode(self, n_samples=1000, return_adj = True, return_logprob=False):
        samples, logprobs = self.sample((n_samples,), return_logprobs=True)
        max_idx = torch.argmax(logprobs)
        mode_ = samples[max_idx].unsqueeze(0)
        if return_adj:
            mode_ = utils.vec_to_adj_mat(mode_.unsqueeze(0), self.n_nodes).squeeze()
        if return_logprob:
            return mode_, logprobs[max_idx]
        return mode_

    @staticmethod
    def _t(a):
        return [t.transpose(0, 1).contiguous() for t in a]