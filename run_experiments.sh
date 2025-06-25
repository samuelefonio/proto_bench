#!/bin/bash

python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -seed 42 > log_lorentz_cifar10_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -seed 117  > log_lorentz_cifar10_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -seed 12345  > log_lorentz_cifar10_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 15 -seed 42 > log_lorentz_cifar10_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 15 -seed 117  > log_lorentz_cifar10_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 15 -seed 12345  > log_lorentz_cifar10_fewdata_15.out 2>&1

python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 30 -seed 42 > log_lorentz_cifar10_fewdata_30.out 2>&1 
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 30 -seed 117  > log_lorentz_cifar10_fewdata_30.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar10.yaml  -ex 30 -seed 12345  > log_lorentz_cifar10_fewdata_30.out 2>&1
#then test the batch