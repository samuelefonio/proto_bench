#!/bin/bash


#poincare

#cifar10

python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 5 -bs 2 -seed 42 -shrink  > log_poincare_cifar10_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 5 -bs 2 -seed 117 -shrink  > log_poincare_cifar10_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 5 -bs 2 -seed 12345 -shrink  > log_poincare_cifar10_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 15 -bs 4 -seed 42 -shrink  > log_poincare_cifar10_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 15 -bs 4 -seed 117 -shrink  > log_poincare_cifar10_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 15 -bs 4 -seed 12345 -shrink  > log_poincare_cifar10_fewdata_15.out 2>&1

python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 30 -bs 4 -seed 42 -shrink  > log_poincare_cifar10_fewdata_30.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 30 -bs 4 -seed 117 -shrink  > log_poincare_cifar10_fewdata_30.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar10.yaml  -ex 30 -bs 4 -seed 12345 -shrink  > log_poincare_cifar10_fewdata_30.out 2>&1

#cifar100

python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 42 -shrink  > log_poincare_cifar100_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 117 -shrink  > log_poincare_cifar100_fewdata_5.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 12345 -shrink  > log_poincare_cifar100_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 42 -shrink  > log_poincare_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 117 -shrink  > log_poincare_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 12345 -shrink  > log_poincare_cifar100_fewdata_15.out 2>&1

python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs 16 -seed 42 -shrink  > log_poincare_cifar100_fewdata_30.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs 16 -seed 117 -shrink  > log_poincare_cifar100_fewdata_30.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs 16 -seed 12345 -shrink  > log_poincare_cifar100_fewdata_30.out 2>&1