import torch
import utils

class FactorisedBase(torch.nn.Module):

    def __init__(self, n_nodes, temp_rsample=0.1, device = 'cpu'):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim_out = n_nodes* (n_nodes-1)
        self.temp_rsample = temp_rsample
        self.device = device

        base_log_probs = torch.randn(self.n_dim_out,1).clone().detach().requires_grad_(True).to(device)

        self.params = torch.nn.Parameter(base_log_probs)

    def forward(self, inputs):
        dist = torch.distributions.Bernoulli(logits=self.params)
        log_prob = dist.log_prob(inputs).sum(dim = (1))
        return log_prob

    def sample(self, size, return_logprobs=False):
        samples, logprobs = self._sample(size, reparametrized=False)
        if return_logprobs:
            return samples, logprobs
        else:
            return samples

    def rsample(self, size, return_logprobs=False, temp=None, hard=False):
        samples, logprobs = self._sample(size, reparametrized=True, temp=temp, hard=hard)
        if return_logprobs:
            return samples, logprobs
        else:
            return samples

    def _sample(self, batch_size, reparametrized=False, temp=None, hard=False):

        if temp is None:
            temp = self.temp_rsample

        if reparametrized:
            _sample = torch.distributions.RelaxedBernoulli(temperature=temp, logits=self.params).rsample(batch_size)
        else:
            _sample = torch.distributions.Bernoulli(logits=self.params).sample(batch_size)
            
        samples = _sample
        if reparametrized and hard:
            samples_hard = utils.one_hot(torch.argmax(_sample, -1), 2)
            samples = (samples_hard - samples).detach() + samples
        
        logprobs = self.forward(samples)
        return samples, logprobs


    def log_prob(self, value):
        if len(value.shape) < 3:
            value = value.unsqueeze(-1)
        return self.forward(value)

    def entropy(self, n_samples=10 ** 6):
        bs = 100000
        curr = 0
        ent = 0.
        while curr < n_samples:
            curr_batch_size = min(bs, n_samples-curr)
            ent -= torch.sum(self.log_prob(self.sample([curr_batch_size])))
            curr += curr_batch_size
        return ent/n_samples


    def mode(self, n_samples=1000, return_adj = True, return_logprob = False):
        samples, logprobs = self.sample((n_samples,), return_logprobs=True)
        max_idx = torch.argmax(logprobs)
        mode_ = samples[max_idx].unsqueeze(0)
        if return_adj:
            mode_ = utils.vec_to_adj_mat(mode_.unsqueeze(0),self.n_nodes).squeeze()
        if return_logprob:
            return mode_, logprobs[max_idx]
        return mode_