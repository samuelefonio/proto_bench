#!/bin/bash

nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0005 -wd 0.0005 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.0005 -wd 0.0003 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.001 -wd 0.0005 -> log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -lr 0.001 -wd 0.001 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &