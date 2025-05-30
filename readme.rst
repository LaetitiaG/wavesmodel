=======
wavesmodel
=======
 
These scripts aims to provide a generative model of traveling waves within retinotopic areas. 
We show here (Grabot et al., 2025, PLOS Computational Biology) a proof-of-concept that this encoding approach is able to characterize local traveling waves in MEG-EEG data.
Traveling waves are first modeled using traveling waves equation mapped on the primary visual cortex. The simulated brain activity is then projected onto the sensors (MEG and EEG). The predicted activity within sensors can then be compared to empirical data. Different models testing different hypotheses on the propagation of neural activity (e.g. with different temporal or spatial frequency, or a specific direction) can be tested against each other.

Getting started
------------------------------------
Depending on which package manager you use, the exact list of required packages can be found:

- in the *environment.yml* file if you are using conda
- in the *pyproject.toml* file if you are using poetry

This code mostly build on `mne-python <https://mne.tools/stable/index.html>`_ version 1.6.0. The code was developed and tested only on Windows.

You can run the minimal_example script to run the full pipeline. Beforehand, you will need to download an example dataset stored on Zenodo (doi: 10.5281/zenodo.13968952).

Cite this work
------------------------------------
If you use this code in your projects, please cite the following reference:

*Grabot L, Merholz G, Winawer J, Heeger DJ, Dugué L (2025) Traveling Waves in the Human Visual Cortex: an MEG-EEG Model-Based Approach. PLOS Computational Biology 21(4): e1013007. https://doi.org/10.1371/journal.pcbi.1013007*

Acknowledgments
------------------------------------
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 852139 - Laura Dugué). 
