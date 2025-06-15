#!/bin/bash


nohup python3 main.py -config configs/config_hps_cub.yaml -seed 42 > log_hps_cub_42.out 2>&1
nohup python3 main.py -config configs/config_hps_cub.yaml -seed 12345 > log_hps_cub_12345.out 2>&1
nohup python3 main.py -config configs/config_hps_cub_sgd.yaml -seed 42 > log_hps_cub_sgd_42.out 2>&1
nohup python3 main.py -config configs/config_hps_cub_sgd.yaml -seed 117 > log_hps_cub_sgd_117.out 2>&1
nohup python3 main.py -config configs/config_hps_cub_sgd.yaml -seed 12345 > log_hps_cub_sgd_12345.out 2>&1