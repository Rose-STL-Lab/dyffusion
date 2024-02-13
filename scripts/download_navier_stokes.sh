#! /bin/bash

# Change the directory to the script's directory, if it is not already
cd "$(dirname "$0")"

# Download the navier-stokes data
bash download_physical_systems_data.sh full navier-stokes-multi
sleep 5
