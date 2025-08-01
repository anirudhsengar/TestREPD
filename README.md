# REPD: Defect Prediction

A machine learning system that predicts software defect likelihood using reconstruction error analysis with autoencoders and statistical distribution fitting.

## 🎯 Overview

This project implements a novel approach to software defect prediction that combines:
- **Autoencoder neural networks** for dimensionality reduction and feature extraction
- **Reconstruction error analysis** to identify anomalous code patterns
- **Statistical distribution fitting** to model defect vs. non-defect characteristics
- **Automated CI/CD integration** via GitHub Actions for real-time code quality assessment

## 🚀 Features

- **Automated defect prediction** on pull requests
- **Traditional software metrics extraction** (complexity, lines of code, etc.)
- **Deep learning-based feature transformation** using autoencoders
- **Probabilistic defect analysis** using Probability Density Functions (PDFs)
- **Pre-trained model support** for fast predictions
- **GitHub Actions integration** for seamless CI/CD workflows

## 📊 How It Works

1. **Feature Extraction**: Extract traditional software metrics from source code.
2. **Model Training**: Train an autoencoder on a large corpus of non-defective code.
3. **Error Analysis**: Calculate the model's reconstruction error for new code. A higher error suggests the code is anomalous.
4. **Distribution Fitting**: Model the reconstruction error distributions for both historically defective and non-defective code.
5. **Prediction**: For new code changes, calculate the PDF value against both distributions to determine which category is a better fit.

## 🔬 Technical Details

### Architecture
- **Autoencoder**: [20, 17, 7] hidden layer configuration
- **Training**: 500 epochs, learning rate 0.001, batch size 128
- **Distributions**: Automatic best-fit selection (e.g., Log-Normal, Pareto) for error distributions.
- **Error Function**: L2 norm reconstruction error

### Supported File Types
- C/C++ source files (`.c`, `.cpp`, `.cxx`, `.cc`, `.h`, `.hpp`, `.hxx`)


## 📊 Example Output

```
📊 Bug Risk Analysis

### 🔄 BEFORE PR Changes:
🎯 Bug Prediction Analysis

File: example.cpp
pdf(Defective | Reconstruction Error): 2.51234e-08
pdf(Non-Defective | Reconstruction Error): 8.94561e-09

### ✅ AFTER PR Changes:
🎯 Bug Prediction Analysis

File: example.cpp
pdf(Defective | Reconstruction Error): 1.12345e-08
pdf(Non-Defective | Reconstruction Error): 7.65432e-08

### 💡 Interpretation:
- The model provides Probability Density Function (PDF) values, not absolute probabilities.
- For each file, compare the `pdf(Defective)` and `pdf(Non-Defective)` values. The higher value indicates the model's classification.
- In the "AFTER" example, the `pdf(Non-Defective)` value is now significantly higher, suggesting the PR changes made the code look less like historically defective files.
```

## 📚 Citation

This implementation is based on the research paper:

**Petar Afric, Lucija Sikic, Adrian Satja Kurdija, Marin Silic**  
*REPD: Source code defect prediction as anomaly detection*  
Journal of Systems and Software, Volume 168, 2020, 110641  
ISSN 0164-1212  
https://doi.org/10.1016/j.jss.2020.110641