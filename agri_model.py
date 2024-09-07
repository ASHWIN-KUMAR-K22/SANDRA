import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

# Load the CSV file

file_path =(r"D:\csv_files\Crop_recommendation.csv")
df = pd.read_csv(file_path)

# Encode the crop labels
df['label'] = df['label'].astype('category').cat.codes

# Split the data into features and target variables
X = df[['label']]
y = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multi-output regression model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'crop_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

# Plot feature importances for one of the regressors
feature_importances = model.estimators_[0].feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), ['label'])
plt.title('Feature Importance')
plt.show()

# Plot actual vs. predicted values for the test set
y_pred = model.predict(X_test)

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for i, column in enumerate(columns):
    ax = axs[i//3, i%3]
    ax.scatter(y_test[column], y_pred[:, i], alpha=0.3)
    ax.plot([y_test[column].min(), y_test[column].max()], [y_test[column].min(), y_test[column].max()], 'k--', lw=3)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Actual vs Predicted {column}')

plt.tight_layout()
plt.show()
