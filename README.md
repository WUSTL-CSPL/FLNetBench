# flnet-bench

## Description 
**FLNetBench**, a PyTorch based federated learning simulation framework, created for experimental research in a paper accepted by [DAC 2025](https://www.dac.com/)

## Installation 
To install flnet-bench, first clone the respository.
    
Next, install [Anaconda](https://www.anaconda.com/products/individual). 

### Configuring each submodule
    
To configure the flnet-bench submodule, from the root directory, run
    
    cd flnet-bench
    conda env create -f environment.yml
    conda activate flnet-bench
    
 The above commands create and activate an Anaconda enviroment, with all of the dependencies needed to run flnet-bench. 

 To configure the ns3-fl-network submodule, from the root directiory, run
    
    cd ns3-fl
    ./waf configure
    ./waf
 
### Simulation 
To start a simulation, from the root directory, run 

    cd flnet-bench
    python3 run.py --config=configs/config.json 
    

There are examples of how to set up the configuration file in flnet-bench/configs/flnet-bench.
