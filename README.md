# Studying motifs in connectome models
This repository contains the code for the analisys performed in the EPFL Master's project 
"Studying motifs in connectome models" carried out in the fall semester 2020/2021.
## Installation
The repository only works in a Linux environement. To download the repository, run <br>
`git clone https://github.com/matsantoro/counting_motifs` <br>

Then move to the downloaded repository and run <br>
`conda install -r requirements.txt` <br>

The code needs the original Flagser software installed to run. Please follow the instructions [here](https://github.com/luetge/flagser)
to install it. After you have installed Flagser, make sure that it is added to the shell path so that you can run commands such as
`flagser-count file_in file_out` <br>

**Warning!** The code in the repository accesses the command line of the os to call `flagser-count`. Make sure this is safe 
for you.
## Data
Data for the analysis is not included in the repository.
### Downloading connectivity data
Connectivity data was obtained from the official BBP website, 
[here](https://bbp.epfl.ch/nmc-portal/downloads), under the connectome section.
Data should be extracted and arranged in the following way:
```
counting_motifs
    data
        original
            average
                cons_locs_pathways_mc0_Column.h5
                cons_locs_pathways_mc1_Column.h5
                ...
            individuals
                pathways_P14-13
                    pathways_mc0_Column.h5
                    pathways_mc1_Column.h5
                    ...
                pathways_P14-14
                ...
```
### Preparing files
To prepare the files for counting, the algorithm needs to run Flagser on the data. 
To do so, we first turn the data into a readable pickle, generate an appropriate Flagser file, and run `flagser-count` on it.

You need to use the `Pickleizer` in `robust_motifs/data` to do so. You can find an example use in `counting_scripts`.

The data structure after this operation should be the following:
```
counting_motifs
    data
        original
        ready
            average
                cons_locs_pathways_mc0_Column
                    cons_locs_pathways_mc0_Column.pkl
                    cons_locs_pathways_mc0_Column.flag
                    cons_locs_pathways_mc0_Column-count.h5
                cons_locs_pathways_mc1_Column
                ...
            individuals_1
                pathways_P14-13
                    cons_locs_pathways_mc0_Column
                        cons_locs_pathways_mc0_Column.pkl
                        cons_locs_pathways_mc0_Column.flag
                        cons_locs_pathways_mc0_Column-count.h5
                    cons_locs_pathways_mc1_Column
                    ...
                pathways_P14-14
                ...      
```
### Preparing controls
To prepare the control models, just run the `create_controls.py` script in `counting_scripts`. The data struture will be
the following:
```
counting_motifs
    data
        original
        ready
        controls_1
            adjusted
                seed_0
                   graph.flag
                   graph.pkl
                   graph-count.h5 
                seed_1
                ...
                seed_4
            pathways
            shuffled
            
```
## Chapter 4 - Bisimplices and extended simplices
### Counting motifs
Bisimplices and extended simplices are stored in arrays. We store bisimplices and extended simplices relative to
directed simplices. For each dimension, each motif has two arrays:
- an array of extra neurons, which are the neurons with which a lower dimension directed simplex creates a bisimplex with
- an array of pointers, which expresses which neurons are related to which simplices.
These arrays will be saved in each graph folder.

To run the algorithm to generate this, you need to use the `Processor` class. You can find an example in `counting_scripts/average_rat.py`.
### Plots
Assuming you have the right folder structure, all plots in section 4 in the thesis are generated from scripts contained in `plot_scripts`. 


The folder `images/final` contains all the generated plots for this section.

Scripts should be moved to the repository root folder before being run.
## Chapter 5 - Bidirectional edges in simplices
### Generating bidirectional edge controls
To generate bidirectional edge controls, use the script `bcount_scripts/create_biedge_control.py`. Change the argument to `'underlying'` to obtain the underlying control model.
### Counting bidirectional edges
To generate bidirectional edge count matrices, you can follow the scripts `bcount_scripts/bcounts_bshuffled.py`. Count matrices are stored as pickled dictionaries.
### Plots
Plots which are extensions of the plots of chapter 4 can be found in the `plot_scripts` folder.

Other plots are generated using notebooks.


The folder `images/bcounts` contains all the generated plots for this section.

Both scripts and notebooks need to be moved to the repository root folder before being run.
## Chapter 6 - Activity and structure
Activity data is not publicly available yet.
### Generating correlation matrices
Correlation matrices are generated and stored using `numpy` following the notebook `notebook/Correlation_matrices.ipynb`.
### Computing average correlations
Average correlations in all simplices and in bisimplices can be computed using the scripts in the folder `activity_scripts`.
### Plots
Activity plots are performed only in notebooks.

The folder `images/activity` contains all the generated plots for this section.

Notebooks need to be moved to the repository root folder before being run.