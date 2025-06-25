#!/bin/bash


nohup python3 main.py -config configs_fewdata/config_ecl_aircraft.yaml -seed 42 > log_ecl_aircraft_fewdata_15.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_aircraft.yaml -seed 117 > log_ecl_aircraft_fewdata_15.out 2>&1 &
nohup python3 main.py -config configs_fewdata/config_ecl_aircraft.yaml -seed 12345 > log_ecl_aircraft_fewdata_15.out 2>&1 &
