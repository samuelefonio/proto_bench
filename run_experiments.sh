#!/bin/bash

nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0003 -wd 0.0001 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0003 -wd 0.00005 -> log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0001 -wd 0.0001 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0001 -wd 0.00005 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
