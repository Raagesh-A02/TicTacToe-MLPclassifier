
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")
X = df.drop("BestMove", axis=1)
y = df["BestMove"]

# Encode board values
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col])

# Train MLP model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
model.fit(X, y)

# Save model
with open("tictactoe_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
