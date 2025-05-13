# FLNetBench

##
run experiment with:
```
./exp.bash
```

## MNIST Dataset URLs:
```
("train-images-idx3-ubyte", "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"),
("train-labels-idx1-ubyte", "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"),
("t10k-images-idx3-ubyte", "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"),
("t10k-labels-idx1-ubyte", "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")
```

## About

**FLNetBench**, a PyTorch based federated learning simulation framework, created for experimental research in a paper accepted by [DAC 2025](https://www.dac.com/)


## Installation

To install **FLNetBench**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLNetBench** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `flnet-bench` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `flnet-bench` environment with:

```shell
conda activate flnet-bench
```

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

##### `run.py` flags

* `--config` (`-c`): path to the configuration file to be used.
* `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.

##### `config.json` files
