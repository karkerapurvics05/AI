import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
df = pd.read_csv('Iris.csv')

# Select only two classes for binary classification
df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]

# Select two features for classification and visualization
X = df[['PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize the feature values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define different SVM parameters for optimization
parameters = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Find the best combination of parameters
grid_search = GridSearchCV(
    SVC(),
    parameters,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Display the best parameters
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

# Train the SVM model using the best parameters
model = grid_search.best_estimator_

# Predict the test data
y_pred = model.predict(X_test)

# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("Test Accuracy Percentage:", accuracy * 100, "%")

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()

classes = model.classes_

plt.xticks(range(len(classes)), classes, rotation=20)
plt.yticks(range(len(classes)), classes)

# Display values inside the confusion matrix
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()

# Create a mesh for the SVM decision boundary
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = __import__('numpy').meshgrid(
    __import__('numpy').arange(x_min, x_max, 0.02),
    __import__('numpy').arange(y_min, y_max, 0.02)
)

# Predict the class for each point in the mesh
Z = model.predict(
    __import__('numpy').c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# Create a mesh for the SVM decision boundaryplt.figure(figsize=(9, 6))

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Convert predicted class labels into numeric values
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.where(Z == classes[0], 0, 1)
Z = Z.reshape(xx.shape)

# Convert training labels into numeric values
train_labels = np.where(y_train == classes[0], 0, 1)

# Convert test labels into numeric values
test_labels = np.where(y_test == classes[0], 0, 1)

# Plot the SVM decision boundary
plt.figure(figsize=(9, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=train_labels,
    edgecolors='black',
    label='Training Data'
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=test_labels,
    marker='x',
    s=80,
    label='Test Data'
)

plt.xlabel('Standardized Petal Length')
plt.ylabel('Standardized Petal Width')
plt.title('SVM Decision Boundary')
plt.legend()

plt.show()
