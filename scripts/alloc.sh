#!/bin/bash
# Script to allocate resources

MEMORY="${1:-128}"
echo "Requested ${MEMORY}G of memory"

salloc -N 1 -n 4 --gres=gpu:h200:1 -p mit_preemptable -t 180 --mem "${MEMORY}G"