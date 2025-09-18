import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
df = pd.read_csv('sample_logins.csv')


import json, os, time
from datetime import datetime, timezone
import numpy as np
'''
 Json technical stuff. Not relevant to the model
'''
def save_metrics_json(
    path,
    *,
    n_features,
    y_true,
    scores,
    chosen_k,
    pr_auc,
    y_pred_at_k,
    split_info=None,
    model_info=None,
    data_file="sample_logins.csv",
    extra=None
):
    y_true = np.asarray(y_true).astype(int)
    y_pred_at_k = np.asarray(y_pred_at_k).astype(int)
    tp = int(((y_true == 1) & (y_pred_at_k == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_at_k == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_at_k == 0)).sum())
    fn = int(((y_true == 1) & (y_pred_at_k == 0)).sum())

    prevalence = float((y_true == 1).mean())
    precision_at_k = float(tp / max(chosen_k, 1))
    recall_at_k = float(tp / max((y_true == 1).sum(), 1))

    payload = {
        "run_info": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": os.environ.get("GIT_COMMIT", ""),
            "data_file": data_file,
            "n_samples": int(len(y_true)),
            "n_features": n_features  # fill if you track it
        },

        "data_split": split_info or {},
        "prevalence": prevalence,
        "isolation_forest": model_info or {},
        "thresholding": {
            "strategy": "fixed_top_k",
            "k": int(chosen_k)
        },
        "metrics": {
            "pr_auc": float(pr_auc),
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "confusion_at_threshold": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }
        }
    }

    if extra:
        payload.update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote metrics to {path}")

'''
FILTERS
'''
df_final = df.query('ip_block > 0 and failed_logins_24h >= 0 and 0 <= hour and hour < 24')
df_final = df_final[df_final['country_changed'].isin([0,1])]  #Valid Country Changed
df_final = df_final[df_final['is_new_ip'].isin([0,1])]  #Valid Country Changed
df_final = df_final[df_final['is_new_device'].isin([0,1])]  #Valid Country Changed
df_final = df_final[df_final['label_anomaly'].isin([0,1])]  #Valid Country Changed

print("The distribution of data is given there")
print(df_final.describe())
print("Number of unique users is: " + str(df_final['user_id'].nunique()))

#Didn't put 'country' to not make the detector discriminate based on country on origin.
features = df_final['label_anomaly']
y = df_final['label_anomaly'].astype(int).values
df_final = df_final[['device_type','failed_logins_24h','hour','country_changed','is_new_ip','is_new_device']]
df_final = pd.get_dummies(df_final, columns=['device_type'])
n_features = df_final.shape[1]

'''
#Scaling data

scaler = StandardScaler().fit_transform(df_final)
df_final = pd.DataFrame(data=scaler, columns=df_final.columns)
'''

#Column 'device_type' must be converted to numeric.

'''
#Histogram of hours

plt.subplot(2,1,1)
plt.title('Hour distribution', fontsize=20)
plt.xlabel('Hour')
plt.ylabel('Number of login attempts')
HIST_BINS = np.linspace(0, 23, 24)
plt.hist(df['hour'], bins=HIST_BINS)
plt.show()

#Bar Chart of devices used
ax = df['device_type'].value_counts().plot(kind='bar')
plt.title('Devices used', fontsize=20)
plt.xlabel('Device')
plt.ylabel('Number of login attempts')

plt.show()
'''
print(features.describe())
contamination_ratio =  len(df[df['label_anomaly'] == 1]) / len(df['label_anomaly'])
print("Contamination ratio is :" + str(contamination_ratio))
forest = IsolationForest(random_state=42,contamination= contamination_ratio)
forest.fit(df_final)



k = 10
scores = -forest.decision_function(df_final)
topk_idx = np.argsort(scores)[::-1][:k]
pred = np.zeros_like(y)
pred[topk_idx] = 1



y_pred = forest.predict(df_final)


print("Now we analyze first five rows: from most anomalous and less and rank see what the classifier decided")
print("Record no.4: " + str(df.iloc[3])) # This is the most anomalous data of the first five rows

print("Anomaly predicted: " + str((1-y_pred[3]) / 2) ) # This is the most anomalous data of the first five rows

print("Record no.5: " + str(df.iloc[4])) # This is the most anomalous data of the first five rows

print("Anomaly predicted: " + str((1-y_pred[4]) / 2) ) # This is the most anomalous data of the first five rows

print("Record no.1: " + str(df.iloc[0])) # This is the most anomalous data of the first five rows

print("Anomaly predicted: " + str((1-y_pred[0]) / 2) ) # This is the most anomalous data of the first five rows

print("Record no.2: " + str(df.iloc[1])) # This is the most anomalous data of the first five rows

print("Anomaly predicted: " + str((1-y_pred[1]) / 2) ) # This is the most anomalous data of the first five rows

print("Record no.3: " + str(df.iloc[2])) # This is the most anomalous data of the first five rows

print("Anomaly predicted: " + str((1-y_pred[2]) / 2) ) # This is the most anomalous data of the first five rows


y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

ap = average_precision_score(y, scores)  # PR-AUC
prec, rec, thr = precision_recall_curve(y, scores)


print("\n PR-AUC: " + str(ap))
if k > 0:
    precision_at_k = (y[topk_idx].sum()) / k
    print("Precision@K (K= "+ str(k) + "): "  +str(precision_at_k))
#Precission at k has the disadvantage that it does not say anything of the quality of the detections at the very top, just the overall quality at our threshold.

# Show the 5 most anomalous sessions
top5 = np.argsort(scores)[::-1][:5]
cols_to_show = ['session_id','user_id','country','device_type','failed_logins_24h','hour','country_changed','is_new_ip','is_new_device','label_anomaly']
print("\nTop-5 anomalies:")


output = df.iloc[top5][cols_to_show].assign(score=scores[top5]).sort_values('score', ascending=False)
output = output.reset_index(drop=True)

print(output)
output.to_csv('worst_anomalies.csv')

#Saving the model
joblib.dump(forest, open('model.pkl', 'wb'))

save_metrics_json(
    "metrics.json",
    n_features = n_features,
    y_true=y,
    scores=scores,
    chosen_k=k,
    pr_auc=ap,
    y_pred_at_k=pred,
    split_info={
        "split_type": "time_ordered",
        "train_size": len(y),  # replace if you actually split
        "test_size": 0,
        "group_guard": "user_id"
    },
    model_info={
        "contamination_param": float(contamination_ratio),
        "random_state": 42
    }
)