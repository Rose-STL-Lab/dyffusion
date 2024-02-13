#! /bin/bash

# Change the directory to the script's directory, if it is not already
cd "$(dirname "$0")"

# Download the spring-mesh data
bash download_physical_systems_data.sh full spring-mesh
sleep 10
