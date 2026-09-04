import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
df = pd.read_csv('Iris.csv')

# Select input features and target variable
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a weak classifier using a decision stump
weak_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

# Train the individual weak classifier
weak_model.fit(X_train, y_train)

# Predict using the weak classifier
weak_pred = weak_model.predict(X_test)

# Calculate weak classifier accuracy
weak_accuracy = accuracy_score(y_test, weak_pred)

print("Weak Classifier Accuracy:", weak_accuracy)
print("Weak Classifier Accuracy Percentage:", weak_accuracy * 100, "%")

# Create the AdaBoost ensemble model
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train the AdaBoost model
model.fit(X_train, y_train)

# Predict using the AdaBoost model
y_pred = model.predict(X_test)

# Calculate AdaBoost accuracy
adaboost_accuracy = accuracy_score(y_test, y_pred)

print("\nAdaBoost Accuracy:", adaboost_accuracy)
print("AdaBoost Accuracy Percentage:", adaboost_accuracy * 100, "%")

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nAdaBoost Confusion Matrix:")
print(cm)

# Display the classification report
print("\nAdaBoost Classification Report:")
print(classification_report(y_test, y_pred))

# Compare the weak classifier and AdaBoost
print("\nPerformance Comparison:")
print("Weak Classifier:", weak_accuracy * 100, "%")
print("AdaBoost:", adaboost_accuracy * 100, "%")

# Visualize the accuracy comparison
models = ['Weak Classifier', 'AdaBoost']
accuracies = [weak_accuracy * 100, adaboost_accuracy * 100]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Weak Classifier vs AdaBoost")
plt.ylim(0, 105)

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%", ha='center')

plt.tight_layout()
plt.show()

# Visualize the AdaBoost confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)
plt.title("AdaBoost Confusion Matrix")
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
