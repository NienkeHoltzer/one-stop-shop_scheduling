# one-stop-shop_scheduling
Robust bi-objective scheduling for a one-stop shop palliative radiotherapy clinic: an extension to flexible flow shop


## Description
Set of scripts as used for the manuscript: "Robust bi-objective scheduling for a one-stop shop palliative radiotherapy clinic: an extension to flexible flow shop"

An experiment is started and initialized via OSS_FindOptimalSolutions.py. This script calls the genetic algorithm (OSS_GeneticAlgorithm) and both MILPs via OSS_SchedulingSolver.py

For the calculation of the risk of overtime, the ProcessingTimes.npy is called, which contains randomly pre-sampled execution times for each task for 5 patients.
