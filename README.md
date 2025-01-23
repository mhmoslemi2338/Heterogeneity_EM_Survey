# Heterogeneity in Entity Matching (HEM)

## Overview

This repository contains the code and supplementary resources for the paper [**"Heterogeneity in Entity Matching: A Survey and Experimental Analysis"**]. Our work examines the challenges posed by data heterogeneity in entity matching (EM), providing a comprehensive survey and experimental analysis. We introduce Heterogeneous EM (HEM), a framework for systematically categorizing and addressing heterogeneity in EM, and evaluate existing methods while proposing future directions for advancing the field.

## Introduction

Entity Matching (EM) is a cornerstone of data management and a critical task for ensuring data accuracy and consistency across disparate sources. Its importance has grown in today’s data-driven world, where effectively linking diverse datasets is essential for generating valuable insights. However, EM becomes particularly challenging in the presence of data heterogeneity, requiring the reconciliation of diverse formats, representations, structures, schemas, and semantics across multiple sources. Addressing this complexity is vital to ensure the reliability and utility of data integration and analysis in increasingly information-rich environments.

Our paper explores EM in heterogeneous data environments, referred to as Heterogeneous EM (HEM), and examines the unique challenges and complexities introduced by heterogeneity. We define data heterogeneity and categorize its various types, distinguishing between representation heterogeneity and semantic heterogeneity. We also analyze HEM through the lens of the FAIR principles—Findability, Accessibility, Interoperability, and Reusability—discussing their impact on heterogeneous data management.

Additionally, we conduct a comprehensive survey of state-of-the-art EM techniques, evaluating their application and effectiveness in handling heterogeneous data. We empirically assess selected EM methods under diverse heterogeneous conditions, with a particular focus on semantic heterogeneity, an area that remains underexplored. Finally, building on our findings, we provide insights into future research directions for advancing HEM.

This repository also addresses fairness in entity matching, proposing a novel score calibration technique that minimizes disparities across different groups. By applying optimal transport-based calibration, we ensure fairness across matching outcomes with minimal loss in model accuracy. Our post-processing approach, leveraging Wasserstein barycenters, is model-agnostic and effective across various state-of-the-art matching methods. Furthermore, to address limitations in reducing Equal Opportunity Difference (EOD) and Equalized Odds (EO) differences, we introduce a conditional calibration method, empirically achieving fairness across widely used benchmarks.

## Key Contributions

- Definition and categorization of Heterogeneous EM (HEM), highlighting challenges posed by data heterogeneity.
- Analysis of HEM through the lens of the FAIR principles, emphasizing their impact on data integration.
- Comprehensive survey of EM methods and their applicability to heterogeneous data environments.
- Empirical evaluation of EM techniques with a focus on semantic heterogeneity.
- Proposal of a novel score calibration method leveraging optimal transport theory for fairness in entity matching.
- Introduction of a conditional calibration method addressing fairness limitations in EOD and EO metrics.

We invite readers to explore the full paper for detailed insights and experimental results.


# Requirements

The following dependencies are required to run the calibration code:

- Python 3.8+
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- SciPy

------
# Data

Obtain the dataset from the DeepMatcher library: [link](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). Place the dataset in the `Dataset` directory. For each dataset, create a new subdirectory inside `Input/Dataset` containing `train.csv`, `valid.csv`, and `test.csv` files.

# Plots

After completing each experiment:
1. Save the output scores.
2. Use the `plot.ipynb` notebook located in each folder to visualize the results.

