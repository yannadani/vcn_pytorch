# Variational Causal Networks
 Pytorch implementation of [Variational Causal Networks: Approximate Bayesian Inference over Causal Structures](https://arxiv.org/abs/2106.07635) (Annadani et al. 2021).
 
[Yashas Annadani](https://yashasannadani.com), [Jonas Rothfuss](https://las.inf.ethz.ch/people/jonas-rothfuss), [Alexandre Lacoste](https://ca.linkedin.com/in/alexandre-lacoste-4032465), [Nino Scherrer](https://ch.linkedin.com/in/ninoscherrer), [Anirudh Goyal](https://anirudh9119.github.io/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/), [Stefan Bauer](https://www.is.mpg.de/~sbauer)
 

## Installation
You can install the dependencies using 
`pip install -r requirements.txt
`

Create Directory structure which looks as follows: `[save_path]/er_1/`

## Examples

Run

`python main.py --num_nodes [num_nodes] --data_seed [data_seed] --anneal --save_path [save_path]`

In the paper we run the model on 20 different data seeds to obtain confidence intervals. If you would like to compare with factorised distribution, run:

`python main.py --num_nodes [num_nodes] --data_seed [data_seed] --anneal --save_path [save_path] --no_autoreg_base`

## Contact

If you have any questions, please address them to: Yashas Annadani `yashas.annadani@gmail.com`



If you use this work, please cite:

	@article{annadani2021variational,
	title={Variational Causal Networks: Approximate Bayesian Inference over Causal Structures},
	author={Annadani, Yashas and Rothfuss, Jonas and Lacoste, Alexandre and Scherrer, Nino and Goyal, Anirudh and Bengio, Yoshua and Bauer, Stefan},
	journal={arXiv preprint arXiv:2106.07635},
	year={2021}
	}
