#!/bin/bash
rm -rf sandbox
rm -rf deepscapy.sif
singularity build -s ./sandbox ./deepscapy.def
singularity build ./deepscapy.sif ./sandbox/
rm -rf sandbox