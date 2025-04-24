# Virtual epilepsy patient cohort
The code used for generating and evaluating synthetic data based on empirical data from 30 patients with drug-resistant epilepsy

The dataset can be found here:

>Dollomaja, B., Wang, H., & Jirsa, V. (2024). *Virtual Epileptic Patient Cohort (v1.0)* [Data set]. **EBRAINS.** https://doi.org/10.25493/XWG2-S8X

The paper describing the dataset: 

>Dollomaja, B., Wang, H. E., Guye, M., Makhalova, J., Bartolomei, F., & Jirsa, V. K. (2025). *Virtual epilepsy patient cohort: generation and evaluation.* **PLOS Computational Biology**, 21(4), e1012911. https://doi.org/10.1371/journal.pcbi.1012911

## Generating synthetic data
The code to generate synthetic spontaneous seizures: `virtual_epileptic_seeg_ret_patient.py`\
The code to generate synthetic stimulated seizures:  `STIM_virtual_epileptic_seeg_ret.py`\
The code to generate synthetic interictal data:      `IIS_virtual_epileptic_seeg_ret.py`\

## Evaluating synthetic data
The code used to compare each modality of synthetic activity to its empirical counterpart:\
`auto_compare_synthetic_empirical.py` (for spontaneous seizures)\
`auto_compare_synthetic_empirical_Stim.py` (for stimulated seizures)\
`IIS_compute_spike_frequency.py` and `IIS_compare_synthetic_empirical.py` (for interictal data)

## Stimulation amplitude and location evaluation
Generating simulated data with varying stimulation amplitude: `STIM_virtual_epileptic_seeg_ret_ControlAmplitude.py`\
Generating simulated data with varying stimulation location: `STIM_virtual_epileptic_seeg_ret_ControlLocation.py`

Comparing empirical and simulated data in such cases: `auto_compare_synthetic_empirical_Stim_ControlAmp.py` and `auto_compare_synthetic_empirical_Stim_ControlLocation.py`

## Generating surrogate data
Surrogate data are generated using random EZ hypothesis: \
`virtual_epileptic_seeg_ret_Control.py`, \
`STIM_virtual_epileptic_seeg_ret_Control.py`,\
`IIS_virtual_epileptic_seeg_ret_Control.py`

## Significance testing
The employed permutation test for statistical significance testing between virtual cohort and surrogate virtual data: `permutation_test_comparison.py`

## Other
Other files are used for managing the structure of the dataset in `data_manager.py` or for plotting figures from the data in `utils_figures` folder.
