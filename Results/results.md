# Quarterly Earnings AI Results

### Hyper paramiters
hidden_layers :	(7, 33, 32, 3)
max_iter : 1000
train_percentage : 0.75
batch_size : 75
solver : adam
activation : relu
learning_rate :	constant
learning_rate_init : 0.001
shuffle :	True

### Cross Val Score Recall at 10 Folds
0.21223776223776225

### Data Split
target
0    955
1    466

### Confusion Matrix
matrix_train :
[713   0]
[326  26]

matrix_test :
[231  11]
[ 97  17]

### Classification Report
Train Classification Report
               precision    recall  f1-score   support

           0       0.69      1.00      0.81       713
           1       1.00      0.07      0.14       352

    accuracy                           0.69      1065
   macro avg       0.84      0.54      0.48      1065
weighted avg       0.79      0.69      0.59      1065

Test Classification Report
report_test :
               precision    recall  f1-score   support

           0       0.70      0.95      0.81       242
           1       0.61      0.15      0.24       114

    accuracy                           0.70       356
   macro avg       0.66      0.55      0.52       356
weighted avg       0.67      0.70      0.63       356

### Permutation Importance
<img width="655" alt="Screenshot 2024-03-23 at 11 17 40 AM" src="https://github.com/joel-1080p/earnings_ai/assets/156847809/865e94ff-6e30-4c49-a36c-47aab888526a">



