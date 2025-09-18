# Login-anomaly-detection
A simple project example of creating a model which detects login anomalies (i.e. logins that different in their properties too much compared to typical logins). Detecting anomalous logins is helpful in identifying targeted cybersecurity attacks.  It uses a toy example as table.
We read the data sample_logins.csv in the Current Working Directory.

As methodology, we use the Isolation Forest unsupervised learning model. The training and testing data are the same in the sense that there is an extra column called 'label_anomaly' which checks if we identified correctly the anomalies. 

The results give high precision (most of detected anomalies are actually anomalies), but small recall (many anomalies however go unnoticed). So, the F-score is also small. 



The parameters and metadata of the model are saved in the models.json file. There is also included metrics about split data in case anyone wishes to do a train-test split.

Works on Python 3.10
