# Prediction of ICU Readmissions Using Data at Patient Discharge using MIMICIII Database  
- - -  

By using this code repository, you can replicate our work. If you are using any part of this code repository, we would appreciate if you cite our paper as follows:   

> *Pakbin, Arash, Parvez Rafi, Nate Hurley, Wade Schulz, M. Harlan Krumholz, and J. Bobak Mortazavi. "Prediction of ICU Readmissions Using Data at Patient Discharge." In 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 4932-4935. IEEE, 2018.*  

### Data
In order to prepare the dataset, you require the access to the [MIMIC-III database](https://mimic.physionet.org/). Then, you need to set up a PostgreSQL database server using steps specified in the MIMIC-III documentation for [windows](https://mimic.physionet.org/tutorials/install-mimic-locally-windows/) or [Unix/Mac](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/) machines.  Our script also curates features from severity-score views[sapsii, sofa, sirs, lods, apsiii, oasis]. Please ensure to add MIMIC-III concepts as specified at the [link](https://github.com/MIT-LCP/mimic-code/tree/master/concepts/severityscores). 

## Phase 1: Data Extraction (SQL, Python)
### Steps to generate required datasets  

1. Clone the repository

       git clone https://github.com/Erakhsha/ICU72hReadmissionMIMICIII  

2. Copy MIMIC-III compressed csv files to the *data/* directory  

3. Execute PostgreSQL scripts available in the */DBScripts* directory on database server. These scripts create required views in the database.

       psql 'dbname=mimic user=xxxx password=xxxx options=--search_path=mimiciii' -f getFeatures_from_labevents.sql  
	   psql 'dbname=mimic user=xxxx password=xxxx options=--search_path=mimiciii' -f getFeatures_from_chartevents.sql  

4. configure database connection strings in the *resources/config.yml* file.  

5. Execute following script to generate datasets

       python generate_datasets/main.py   

## Phase 2: XGBoost (Python)
The second phase trains XGBoost in Python and saves results which include, but are not limited to: ROC plots for each fold and the actual probabilities for each person and the feature ranking among all of the folds for each specific label.  
### Execute scripts in *model1*  
Exucute fold_saver.py located below. There is a small block at the beginning of the script which needs to be set. Please note that for some functionalities in phase 3, you have to set 'save_folds_data' parameter equal to true so functions such as 'Calibration Plot' can work.  

       python models1/fold_saver.py 
	   
## Phase 3:  LR (and XGBoost) and Calibration Plots (R)

The third phase trains LR (and XGBoost) models and calibration plots for models trained in second and third phase. It should be noted that this part is written in R. You need to run scripts located in *models2* folder.  
