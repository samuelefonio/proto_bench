#!/bin/bash

#ecl

# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 5 -bs 8 -seed 42 > log_ecl_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 5 -bs 8 -seed 117  > log_ecl_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 5 -bs 8 -seed 12345  > log_ecl_cifar100_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 15 -bs 16 -seed 42 > log_ecl_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 15 -bs 16 -seed 117  > log_ecl_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 15 -bs 16 -seed 12345  > log_ecl_cifar100_fewdata_15.out 2>&1

# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 30 -bs  -seed 42 > log_ecl_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 30 -bs  -seed 117  > log_ecl_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_ecl_cifar100.yaml  -ex 30 -bs  -seed 12345  > log_ecl_cifar100_fewdata_30.out 2>&1

##############################################################################################################################################

#hps

# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 5 -bs 8 -seed 42  > log_hps_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 5 -bs 8 -seed 117  > log_hps_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 5 -bs 8 -seed 12345  > log_hps_cifar100_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 15 -bs 16 -seed 42  > log_hps_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 15 -bs 16  -seed 117  > log_hps_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 15 -bs 16  -seed 12345  > log_hps_cifar100_fewdata_15.out 2>&1

# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 30 -bs  -seed 42  > log_hps_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 30 -bs  -seed 117  > log_hps_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_hps_cifar100.yaml  -ex 30 -bs  -seed 12345  > log_hps_cifar100_fewdata_30.out 2>&1

##############################################################################################################################################

#lorentz

# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 5 -bs 8 -seed 42  > log_lorentz_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 5 -bs 8 -seed 117  > log_lorentz_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 5 -bs 8 -seed 12345  > log_lorentz_cifar100_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 15 -bs 16 -seed 42  > log_lorentz_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 15 -bs 16 -seed 117  > log_lorentz_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 15 -bs 16 -seed 12345  > log_lorentz_cifar100_fewdata_15.out 2>&1

# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 30 -bs  -seed 42  > log_lorentz_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 30 -bs  -seed 117  > log_lorentz_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_lorentz_cifar100.yaml  -ex 30 -bs  -seed 12345  > log_lorentz_cifar100_fewdata_30.out 2>&1

##############################################################################################################################################

#poincare

# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 42 -shrink -protoopt > log_poincare_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 117 -shrink -protoopt > log_poincare_cifar100_fewdata_5.out 2>&1
# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 5 -bs 8 -seed 12345 -shrink -protoopt > log_poincare_cifar100_fewdata_5.out 2>&1

python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 42 -shrink -protoopt > log_poincare_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 117 -shrink -protoopt > log_poincare_cifar100_fewdata_15.out 2>&1
python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 15 -bs 16 -seed 12345 -shrink -protoopt > log_poincare_cifar100_fewdata_15.out 2>&1

# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs  -seed 42 -shrink -protoopt > log_poincare_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs  -seed 117 -shrink -protoopt > log_poincare_cifar100_fewdata_30.out 2>&1
# python3 main.py -config configs_fewdata/config_poincare_cifar100.yaml  -ex 30 -bs  -seed 12345 -shrink -protoopt > log_poincare_cifar100_fewdata_30.out 2>&1