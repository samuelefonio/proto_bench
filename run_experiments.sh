#!/bin/bash


python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 15 -bs 16 -seed 42 > log_ecl_cifar100_fewdata_15.out 2>&1