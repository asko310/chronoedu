# Importăm librăriile necesare
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Setăm stilul pentru vizualizare
sns.set(style="whitegrid")

# Încărcăm datele
data = pd.read_csv("spectra_data.csv")

# Explorăm datele
print(data.head())

# Împărțim datele în caracteristici (spectru) și etichete (toxicitatea)
X = data.drop(columns=["toxic"])  # Presupunem că ultima coloană este eticheta
y = data["toxic"]

# Împărțim datele în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Creăm și antrenăm modelul
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Facem predicții pe setul de test
y_pred = model.predict(X_test)

# Evaluăm modelul
print("Acuratețea modelului:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Vizualizăm importanța caracteristicilor
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Importanța caracteristicilor")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()

# Salvăm modelul antrenat
joblib.dump(model, "toxic_colorant_detector.pkl")

# Încărcăm modelul pentru a face predicții noi
loaded_model = joblib.load("toxic_colorant_detector.pkl")

# Exemplu de spectru nou cu intensități
new_spectrum = np.array([0.5, 0.8, 1.2, 0.3, 0.4, 0.7]).reshape(1, -1)

# Facem predicția pentru spectrul nou
prediction = loaded_model.predict(new_spectrum)
print("Predicția pentru spectrul nou:", prediction)
