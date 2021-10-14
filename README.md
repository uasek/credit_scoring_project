# Model Risk in Credit Scoring

This repo is aimed at providing quantitative evidence how various stages of constructing a credit scoring model contribute to its performance.

_Financial Risk Management Lab, Higher School of Economics, Moscow, 2021-2022_


## Repo Structure

`/baseline/*` — Baseline (minimum viable product — MVP) model, proof of concept

`/datasets/*` — Overview of 16 most popular datasets employed in the credit scoring literature

`/scoring_model/*` — Modular architecture of all the credit scoring stages

`requirements.txt` lists all the libraries and versions to set up the virtual environment


## Modules Included

Overall pipeline is initially constructing using `sklearn.pipeline.Pipeline()` 
(docs)[https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html].

1. Preprocessing — many approaches already implemented in `feature-engine` library:
	+ missing values imputation
	+ categorical feature encoding
	+ variable transformation
	+ outlier detection and correction
2. Feature Engineering:
	+ featuretools (docs)[https://featuretools.alteryx.com/en/stable/]
	+ Feature-engine — already supports sklearn-like `.fit()`/`.transform()` architecture (docs)[https://feature-engine.readthedocs.io/en/1.1.x/]
3. Feature Selection
4. Additional Modules:
	+ Target imbalance correction
	+ Reject inference
5. Modelling:
	+ Hyperparameter optimization
6. Validation
	+ If trained on subsample: test if the sample is representative (PSI for sample vs. population)
	+ Gini index + confidence interval
	+ Discrimination stability (drops of quality — Train vs. Test OOS vs. Test OOT)
	+ Gini index for each model submodule
	+ Gini dynamics
	+ Factor contribution — Uplift test
	+ PD distribution — check if PDs are not concentrated in a small interval of values
	+ Calibration
	+ Precision-Recall curve stability
	+ Other metrics



