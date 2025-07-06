# ADMET Feature Selection Project

## Overview

This project contains feature selection workflows for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction using ADMETLab3.0 and ADMETSar3.0 platforms.

## Project Structure

The project is organized into two main prediction frameworks, each containing four different datasets:

### ADMETLab3.0 Predictions
- `ADMETLab3.0_pred/FDA_ALL/`
- `ADMETLab3.0_pred/FDA_ChEMBL/`
- `ADMETLab3.0_pred/FDA_GDB17/`
- `ADMETLab3.0_pred/FDA_ZINC/`

### ADMETSar3.0 Predictions
- `ADMETSar3.0_pred/FDA_ALL/`
- `ADMETSar3.0_pred/FDA_ChEMBL/`
- `ADMETSar3.0_pred/FDA_GDB17/`
- `ADMETSar3.0_pred/FDA_ZINC/`

## Directory Contents

Each dataset directory contains:

### Feature Selection Results
- `Mutual_Information_41` through `Mutual_Information_50`: Results from mutual information-based feature selection
- `RandomForestClassifier_41` through `RandomForestClassifier_50`: Random Forest classifier results

### Main Workflow
- `ADMETLab_selection.ipynb`: Primary Jupyter notebook containing the feature selection workflow

## Datasets

The project covers four molecular databases:
- **FDA_ALL**: Complete FDA dataset
- **FDA_ChEMBL**: ChEMBL database subset
- **FDA_GDB17**: GDB17 database subset  
- **FDA_ZINC**: ZINC database subset

## Usage

Navigate to any dataset directory and run the `ADMETLab_selection.ipynb` notebook to execute the feature selection workflow. The numbered directories (41-50) contain the corresponding results for each iteration.