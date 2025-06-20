#!/bin/bash

nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -seed 117 > log_ecl_cifar10_fewdata_5ex_def.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -seed 12345 > log_ecl_cifar10_fewdata_5ex_def.out 2>&1 &