# NGAFID-SSL: Self-Supervised Learning with General Aviation Data

Link to [paper](https://openreview.net/forum?id=6FcBsTcXVE).
## Team
| Name       | Email              |
|------------|--------------------|
| Aidan LaBella  | aidan_labella@brown.edu    |
| Aditya Iyer |  aditya_iyer@brown.edu  |
| Charlie Duong  | charles_duong@brown.edu   |
| Elise Carman  | elise_carman@brown.edu   |
| Nathan DePiero | nathan_depiero@brown.edu |
| Justin Long | pak_iong_long@brown.edu |

## Resources
### Visuals
* [Circuit Diagram of LOC-I Calculations](./LOCI_diagram.pdf)
### Videos
* [Aidan's presentation to ACM SAC '22 on NGAFID data](https://youtu.be/3aDtjYJpVZA)
### Code
* [NGAFID Codebase](https://github.com/travisdesell/ngafid2.0)
* [TST Codebase](https://github.com/gzerveas/mvts_transformer)
* [SimCLR Codebase](https://github.com/sthalles/SimCLR)
### Papers
* [Loss of Control/Stall Index](https://www.se.rit.edu/~travis/papers/2022_sac_ngafid.pdf)
* [Predictive Maintenance](https://arxiv.org/abs/2110.03757)
* [Phase of Flight ID with SOMs](https://www.se.rit.edu/~travis/papers/2024_IEEE_WCCI_ms_som.pdf)
* [(Class Paper) Noisy Timeseries SSL](https://arxiv.org/pdf/2112.10139)
* [Guillotine Regularization](https://arxiv.org/abs/2206.13378)

## Installation

```
$ conda env create --name ngafid-ssl --file ngafid_ssl_environment.yml
$ conda activate ngafid-ssl
$ python run.py
```

## Config file

To change running configurations, pass keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name NGAFID --log-every-n-steps 100 --epochs 100 

```
To run on CPU use the ```--disable-cuda``` option.
For 16-bit precision GPU training, use the ```--fp16_precision``` flag. This will use [Pytorch built in AMP training](https://pytorch.org/docs/stable/notes/amp_examples.html).
