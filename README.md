# PANOPTIC-GRAPHICAL-REGRESSION

## 🎯 Overview  
`PANOPTIC-GRAPHICAL-REGRESSION` is a Python implementation for performing graphical regression and analysing detection performance metrics (e.g., \(P_d\) vs \(P_{fa}\), and \(P_d\) vs correlation coefficient). The aim is to replicate published performance plots and enable flexible experimentation with detection/regression workflows.

## 🧮 Key Features  
- Loads and processes result data (e.g., `gait_results.pkl`).  
- Computes detection performance metrics and generates plots (such as \(P_d\) vs \(P_{fa}\), \(P_d\) vs correlation).  
- Provides modular code for main execution (`main.py`) and performance analysis (`perf.py`).  
- Easy to extend for new data sets, detection thresholds, or regression models.

## 📁 Repository Structure  

├── gait_results.pkl ← Sample result data file

├── main.py ← Entry-point for running the experiment

├── perf.py ← Performance analysis & plotting utilities

└── README.md ← (You are here)


## 🛠 Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/lavanyasusanna54/PANOPTIC-GRAPHICAL-REGRESSION.git
   cd PANOPTIC-GRAPHICAL-REGRESSION

