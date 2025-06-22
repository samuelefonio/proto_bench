#!/bin/bash


nohup python3 main.py -config configs_fewdata/config_hps_cifar10.yaml -lr 0.0003 -wd 0.0003 > log_hps_cifar10_fewdata_5.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_hps_cifar10.yaml -lr 0.001 -wd 0.001 > log_hps_cifar10_fewdata_5.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_hps_cifar10.yaml -lr 0.001 -wd 0.0003 > log_hps_cifar10_fewdata_5.out 2>&1 &