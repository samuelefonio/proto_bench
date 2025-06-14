#!/bin/bash


nohup python3 main.py -config configs/config_ecl_cifar10_01_12345.yaml > log_ecl_cifar10_01_12345.out 2>&1
nohup python3 main.py -config configs/config_ecl_aircraft_1_117.yaml > log_ecl_aircraft_1_117.out 2>&1
nohup python3 main.py -config configs/config_ecl_aircraft_1_12345.yaml > log_ecl_aircraft_1_12345.out 2>&1