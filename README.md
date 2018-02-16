# Prediction of ICU Readmissions Using Data at Patient Discharge using MIMICIII Database  
- - -  

By using this code repository, you can replicate our work. If you are using any part of this code repository, we would appreciate if you cite our paper as follows:   

> *"citation"*  

### Data
In order to prepare the dataset, you are required to acquire access to the [MIMIC-III database](https://mimic.physionet.org/). Thereafter, you are required to set up a PostgreSQL database server using steps specified in the MIMIC-III documentation for [windows](https://mimic.physionet.org/tutorials/install-mimic-locally-windows/) or [Unix/Mac](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/) machines.


In the first phase of the scripts, data is extracted, some part of which is in SQL and the remaining part is in Python. 

The second phase is training XGBoost in Python and saving results which include but are not limited to: ROC plots for each fold and the actual probabilities for each person and the feature ranking among all of the folds for each specific label.

The third phase is training LR (and XGBoost) models and calibration plots for models trained in second and third phase. It should be noted that this part is written in R.

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
  
### Steps to train and test the models  