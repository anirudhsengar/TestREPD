# REPD: Defect Prediction

A machine learning system that predicts software defect probability using reconstruction error analysis with autoencoders and statistical distribution fitting.

## 🎯 Overview

This project implements a novel approach to software defect prediction that combines:
- **Autoencoder neural networks** for dimensionality reduction and feature extraction
- **Reconstruction error analysis** to identify anomalous code patterns
- **Statistical distribution fitting** to model defect vs. non-defect probability distributions
- **Automated CI/CD integration** via GitHub Actions for real-time code quality assessment

## 🚀 Features

- **Automated defect prediction** on pull requests
- **Traditional software metrics extraction** (complexity, lines of code, etc.)
- **Deep learning-based feature transformation** using autoencoders
- **Probabilistic risk scoring** (0-100 scale)
- **Pre-trained model support** for fast predictions
- **GitHub Actions integration** for seamless CI/CD workflows

## 📊 How It Works

1. **Feature Extraction**: Extract traditional software metrics from source code
2. **Model Training**: Train an autoencoder on historical defect data
3. **Error Analysis**: Calculate reconstruction errors for new code
4. **Distribution Fitting**: Model error distributions for defective vs. non-defective code
5. **Prediction**: Generate probability-based risk scores for new code changes

## 🔬 Technical Details

### Architecture
- **Autoencoder**: [20, 17, 7] hidden layer configuration
- **Training**: 500 epochs, learning rate 0.001, batch size 128
- **Distributions**: Automatic best-fit selection (Log-Normal, Pareto, etc.)
- **Error Function**: L2 norm reconstruction error

### Supported File Types
- C/C++ source files (`.c`, `.cpp`, `.cxx`, `.cc`, `.h`, `.hpp`, `.hxx`)


## 📊 Example Output

```
🎯 Code Quality Risk Assessment

### 🔄 BEFORE PR Changes:
File: example.cpp
- Risk Score: 23.4/100

### ✅ AFTER PR Changes:  
File: example.cpp
- Risk Score: 18.7/100

### 💡 Interpretation:
- Higher risk scores indicate increased likelihood of bugs
- Lower risk scores suggest improved code quality
- Compare BEFORE vs AFTER to see the impact of your changes
```

## 📚 Citation

This implementation is based on the research paper:

**Petar Afric, Lucija Sikic, Adrian Satja Kurdija, Marin Silic**  
*REPD: Source code defect prediction as anomaly detection*  
Journal of Systems and Software, Volume 168, 2020, 110641  
ISSN 0164-1212  
https://doi.org/10.1016/j.jss.2020.110641