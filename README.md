# Metropolis algorithm in CUDA GPGPU framework 

This is an implementation of Metropolis algorithm for Ising model on kagome lattice. 

## Prerequisities 
 
 * [CUDA](https://developer.nvidia.com/cuda-downloads) (obviously)
 * [CUB](https://nvlabs.github.io/cub/) library

## Features
 
 * Calculates energy and magnetizations
 * Calculates and saves mean values of observables (if `CALCULATE_MEANS` macro is defined)
 * Saves the final configuration of the lattice, if `SAVE_CONFIGURATION` macro is defined
 * Saves time series of observables, if `SAVE_TS` macro is defined
 * Random numbers are produced in batch for better performance
 * Logic of Metropolis algorithm is separated from the lattice shape (contained in kernels) 

