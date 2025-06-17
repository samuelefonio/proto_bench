#!/bin/bash

nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -bs 4 > log_ecl_cifar10_fewdata_5ex_42_bs4.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -bs 2 > log_ecl_cifar10_fewdata_5ex_42_bs2.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -bs 4 -lr 0.00005 > log_ecl_cifar10_fewdata_5ex_42_bs4.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_cifar10.yaml -bs 4 -lr 0.00001 > log_ecl_cifar10_fewdata_5ex_42_bs4.out 2>&1 &
