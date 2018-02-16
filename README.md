# ICU72hReadmissionMIMICIII
Prediction of ICU Readmissions Using Data at Patient Discharge using MIMICIII Database
======================================================================================  

By using this code repository, you can replicate our work. If you are using any part of this code repository, we would appreciate if you cite our paper as follows:   

> "<< CITATION >>".  

### Data
Before executing the scripts, you need to acquire access to the [MIMIC-III database] (https://mimic.physionet.org/). Thereafter, you need to set up a PostgreSQL database server using steps specified in the MIMIC-III documentation for [windows](https://mimic.physionet.org/tutorials/install-mimic-locally-windows/) or [Unix/Mac](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/) machines.


In the first phase of the scripts, data is extracted, some part of which is in SQL and the remaining part is in Python. 

The second phase is training XGBoost in Python and saving results which include but are not limited to: ROC plots for each fold and the actual probabilities for each person and the feature ranking among all of the folds for each specific label.

The third phase is training LR (and XGBoost) models and calibration plots for models trained in second and third phase. It should be noted that this part is written in R.

### Steps to generate required dataset  

Step 1: Clone the repository  
https://github.com/Erakhsha/ICU72hReadmissionMIMICIII  

Step 2: Copy MIMIC-III compressed csv files to the data/ directory  

Step 3: Execute PostgreSQL scripts available in the /DBScripts directory on database server. These scripts create required views in the database. 
psql 'dbname=mimic user=XXXXXX password=XXXXXX options=--search_path=mimiciii' -f getFeatures_from_labevents.sql  
psql 'dbname=mimic user=XXXXXX password=XXXXXX options=--search_path=mimiciii' -f getFeatures_from_chartevents.sql  

Step 4: configure database connection strings in the resources/config.yml file.


### Steps to train and test the models