# ICU72hReadmissionMIMICIII
Prediction of ICU Readmissions Using Data at Patient Discharge using MIMICIII Database
======================================================================================

By using this code, you can replicate our work which is ICU readmission prediction using MIMICIII database.
The first phase is the data extraction some part of which is in SQL and the remaining part is in Python. We have used SQL because Chartevents and Labevents tables are too large to be managed by Python.
The second phase is training XGBoost in Python and saving results which include but are not limited to: ROC plots for each fold and the actual probabilities for each person and the feature ranking among all of the folds for each specific label.
The third phase is training LR (and XGBoost) models and calibration plots for models trained in second and third phase. It should be noted that this part is written in R.

If you are using this code, please cite our paper :

Regards
Arash
