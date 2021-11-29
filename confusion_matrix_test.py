from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

# y_true = [[1], [1,2], [1,2,3]]
# y_pred = [[2], [4,5], [1,5,6]]

y_true = [1,2,3,4]
y_pred = [4,5,6,2]

print(confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6]) + confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6]))