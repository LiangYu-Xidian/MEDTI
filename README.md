# MEDTI
Integrating multi-layer networks for Drug-Target Interaction Prediction

## Dependencies
*MEDTI* is tested to work under Python 3.7 and Matlab 9.4.

## Code and Data
#### Code
- `DTINet.m`: predict drug-target interactions (DTIs)
- `run_DTINet.m`: example code of running `DTINet.m` for drug-target prediction
- `train_mf.mexa64`: pre-built binary file of inductive matrix completion algorithm (downloaded from [here](http://bigdata.ices.utexas.edu/software/inductive-matrix-completion/))
- `main.py`: Learning feature vectors using multi-layer network representation learning models
- `MEDTI.py`: Building a multi-layer network representation learning model
- `preprocessing.py`: Similarity network data preprocessing
- `MEDTI_params.txt`: Parameters required to run main.py function

#### Data: `test_data/` directory
- `mat_drug_protein.txt` 	    	: Drug_Protein interaction matrix
- `drug/drug_common.txt`       		: list of drug IDs
- `drug/Sim_drug_se.txt` 			: Drug-SideEffect Similarity data
- `drug/Sim_drug_drug.txt` 			: Drug-Drug interaction data
- `drug/Sim_drug_disease.txt` 		: Drug-Disease Similarity data
- `drug/Sim_drug_strc.txt` 	    	: Drug chemical structure Similarity data
- `protein/protein_common.txt`  	: list of protein IDs
- `protein/Sim_protein_seq.txt` 	: Protein similarity scores based on primary sequences of proteins
- `protein/Sim_protein_protein.txt`	: Protein-Protein interaction data
- `protein/Sim_protein_disease.txt` : Protein-Disease Similarity data
**Note**: drugs, proteins, are organized in the same order across all files, including ID lists, and Similarity data.

### Result data
#### `test_data/test_results` directory
We provided the pre-trained vector representations for drugs and proteins, which were used to produce the results in our paper.
- `drug_arch_2_1500_features.txt`		: Drugs feature
- `protein_arch_2_2500_features.txt`	: Proteins feature
- `prediction.txt`	: The prediction result of drug-target interactions

#### `test_data/test_models` directory
Stored in the model to learn the drug and protein feature vectors, the loss change chart.

#### `test_data/mat_data` directory
We preprocess the similarity data of drugs and proteins to obtain mat files.

### Tutorial
1. Put drug and protein similarity data in the `test_data/drug` and `test_data/protein` folder.
2. Create a `mat_data/` folder under `MEDTI/test_data/` and run `preprocessing.py`, which will compute the mat data of drugs and proteins similarity data.
3. Create a `test_results/` and `test_models/` folder under `MEDTI/test_data/` and run `main.py`, which will compute the feature of drugs and proteins.
4. Run `run_DTINet.m`, this script will predict the drug-target interactions and evaluate the results using a ten-fold cross-validation.

### Contacts
If you have any questions or comments, please feel free to email Liang Yu (lyu@xidian.edu.cn) and/or Yifan Shang (shangyifan123@126.com).
