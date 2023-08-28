# README

This directory contains code for reproducing experiments in the paper
> Narges Ahani, Paul Gölz, Ariel Procaccia, Alexander Teytelboym, and Andrew C. 
> Trapp. Dynamic placement in refugee resettlement. Operations Research (2023).

For a description of the dataset, data preprocessing, and employment
predictions, see Section 1 of the e-compendium.

## Script Structure
The main script for running experiments is `compare_matching_algorithms.py`.
It simulates multiple refugee allocation algorithms on different arrival
scenarios, and writes the result to a file given to it in the `trace` argument.
More details on usage can be obtained by running
`python compare_matching_algorithms.py --help`.

The results of this experiment can then be visualized with `plotting.py`.

The file `duals_by_arrival_numbers.py` runs and visualizes experiments
analogous to Figure EC.6.

Finally, the script `time_many_bootstrapped_arrivals.py` measures the running
time of our algorithms on simulated (bootstrapped) sequences of arrivals of
arbitrary length.

## Data Not Shared
Due to a non-disclosure agreement with our data source, we are unable to share
some of the data underlying our experiments. In this section, we give a quick
overview over these omitted data files:

### Refugee Arrivals
The main data we cannot share is the data file detailing the sequence of cases
allocated by HIAS, along with information about many sensitive attributes of
the refugee. Our code reads this information from a file`~/HIAS_FY06-FY20.csv`,
which is omitted in this repository. For a description of these features, see
EC.1 in
[the supplementary material of Ahani et al. (2021)](https://pubsonline.informs.org/doi/suppl/10.1287/opre.2020.2093/suppl_file/opre.2020.2093.sm1.pdf).

### Regression Weights
We are also unable to share the exact weights of the regression used to predict
refugees' employment probabilities. As described in our e-compendium, our first
regression follows the methodology of Ahani et al. (2021), but is trained on a
slightly larger time frame of arrivals. As in Ahani et al. (2021), this
regression is only trained on refugees without US ties, which is why we refer
to it as “NUST” (=**n**o **US** **t**ies). In addition, we train a second
regression (“UST”) on agents with US ties. We cannot share these regression
weights out of a concern of identifying individual HIAS affiliates.
Affiliated-anonymized regression weights for the original timeframe and
refugees without US ties can be found in table EC.2 of the
[e-companion of Ahani et al. (2021)](https://pubsonline.informs.org/doi/suppl/10.1287/opre.2020.2093/suppl_file/opre.2020.2093.sm1.pdf)
This code base expects the regression weights to be given in the files
`annie_ml/ML_outputs/nust_models.pkl` and
`annie_ml/ML_outputs/NUST_feature_coefs.csv` (for the NUST regression), and in
`annie_ml/ML_outputs/ust_models.pkl` and
`annie_ml/ML_outputs/UST_feature_coefs.csv` (for the UST regression).

Closely related to this is the file `annie_ml/ML_outputs/synergy_list.txt`,
which simply contains, line-per-line, the 16 affiliates for which the
regressions insert dummy values and interactions.

### Affiliate Compatibility Information
We cannot share which HIAS affiliates have which restrictions in terms
of which cases they can serve. This information would be provided in
`annie_ml/compatibilities.csv`, a CSV table with one row for each affiliate.
For each affiliate, this table lists which nationalities and languages this
affiliate can serve, whether it can host cases with more than 5 members, and
whether it can host single parents.

### Exact Affiliate Capacities
At the top of `optimization/capacities.py`, a dictionary called `raw_caps`
lists the exact capacities of all affiliates in all fiscal years we study,
including major revisions if they took place in the fiscal year. To protect
HIAS data, this information is removed in this repository; instead, only a
dummy dictionary is defined that demonstrates the overall shape of the data.

# Additional Notes
## Unemployment Rates
The file `annie_ml/data/macrodata.csv` lists the employment rates, over time,
for the counties in which HIAS affiliates are located. This data can be freely
obtained from the US Bureau of Labor Statistics (and is in the public domain),
but we provide this information for easier reproducibility of our results.

## References
- Ahani, Andersson, Martinello, Teytelboym & Trapp. Placement Optimization in Refugee Resettlement. Operations Research (2021) doi:10/gj3kc6.
