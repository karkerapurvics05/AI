import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Iris.csv')

print("First five records:")
print(df.head())

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

model = DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap='Blues')

plt.title(
    "Decision Tree Confusion Matrix",
    fontsize=14,
    fontweight='bold',
    color='darkblue'
)

plt.xlabel("Predicted Class", fontsize=11, color='darkblue')
plt.ylabel("Actual Class", fontsize=11, color='darkblue')

plt.colorbar()

classes = model.classes_

plt.xticks(
    range(len(classes)),
    classes,
    rotation=30,
    color='darkblue'
)

plt.yticks(
    range(len(classes)),
    classes,
    color='darkblue'
)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            color='black'
        )

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False
)

plt.title(
    "Generated Decision Tree",
    fontsize=16,
    fontweight='bold',
    color='darkgreen'
)

plt.tight_layout()
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))

colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

plt.bar(
    importance.index,
    importance.values,
    color=colors,
    edgecolor='black'
)

plt.xlabel(
    "Features",
    fontsize=11,
    fontweight='bold',
    color='darkblue'
)

plt.ylabel(
    "Importance",
    fontsize=11,
    fontweight='bold',
    color='darkblue'
)

plt.title(
    "Feature Importance in Decision Tree",
    fontsize=14,
    fontweight='bold',
    color='darkblue'
)

plt.xticks(rotation=20, color='darkblue')
plt.yticks(color='darkblue')

for i, value in enumerate(importance.values):
    plt.text(
        i,
        value + 0.01,
        f'{value:.3f}',
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()
