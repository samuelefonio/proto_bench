#!/bin/bash

python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml  -ex 30 -bs 2 -seed 42 > log_ecl_cifar10_fewdata_30.out 2>&1 &
python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml  -ex 30 -bs 4 -seed 42 > log_ecl_cifar10_fewdata_30.out 2>&1 &