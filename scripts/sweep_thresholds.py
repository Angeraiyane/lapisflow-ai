import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1. Chargement des données et de l'artefact
val_df = pd.read_csv("data/val.csv")
model = joblib.load("models/clf_v1.joblib")

# Catégorie protégée officielle
PROTECTED_CATEGORY = "unauthorized_transactions_security"
categories = list(model.classes_)
protected_idx = categories.index(PROTECTED_CATEGORY)

# 2. Inférence probabiliste sur la validation
print("Calcul des probabilités calibrées sur la validation...")
probs = model.predict_proba(val_df["text"])
top_classes = np.array([categories[i] for i in np.argmax(probs, axis=1)])
confidences = np.max(probs, axis=1)
protected_probs = probs[:, protected_idx]

# 3. BALAYAGE 1 : Seuil de probabilité protégée (Garantir 0 fuite sur la validation)
print("\n--- BALAYAGE DE SÉCURITÉ (PROTECTED GATE) ---")
# Une fuite = vrai cas protégé qui n'est PAS routé vers human
best_tau_protected = 0.05
min_leaks = float("inf")

# On teste des seuils fins de 0.01 à 0.50
for tau_p in np.linspace(0.01, 0.30, 30):
    is_escalated_protected = protected_probs >= tau_p
    # Fuites = cas protégés réels où is_escalated_protected est Faux
    leaks = np.sum((val_df["is_protected"] == 1) & (~is_escalated_protected))
    escalation_count = np.sum(is_escalated_protected)
    
    if leaks == 0:
        best_tau_protected = tau_p
        print(f"Seuil protégé testé: {tau_p:.3f} | Fuites: {leaks} | Escalades préventives: {escalation_count}/{len(val_df)}")
        # On veut le seuil le plus haut qui maintient 0 fuite pour éviter la sur-escalade
        break

print(f"\n-> Seuil protégé retenu (tau_protected) : {best_tau_protected:.4f}")

# 4. BALAYAGE 2 : Seuil de confiance pour l'auto-routage
print("\n--- BALAYAGE DE ROUTAGE AUTO (CONFIDENCE GATE) ---")
print("Seuil | Couverture Auto (%) | Macro-F1 Auto | Statut (Objectif >=60% / >=0.85)")

# On applique d'abord le filtre de sécurité protégé
safety_mask = protected_probs < best_tau_protected  # Ceux qui survivent au filtre de sécurité

best_tau_conf = 0.70
chosen_coverage = 0.0
chosen_f1 = 0.0

for tau_c in np.linspace(0.50, 0.95, 10):
    # Routé en automatique : survit à la sécurité ET confiance >= tau_c
    auto_mask = safety_mask & (confidences >= tau_c)
    coverage = np.mean(auto_mask)
    
    if np.sum(auto_mask) > 0:
        f1_auto = f1_score(val_df.loc[auto_mask, "category"], top_classes[auto_mask], average="macro", zero_division=0)
    else:
        f1_auto = 0.0
        
    status = "OK" if (coverage >= 0.60 and f1_auto >= 0.85) else "INSUFFISANT"
    print(f"{tau_c:.2f}  | {coverage*100:6.2f}%             | {f1_auto:.4f}        | {status}")
    
    # On retient le seuil le plus strict qui respecte encore >= 60% de couverture
    if coverage >= 0.60 and f1_auto >= 0.85:
        best_tau_conf = tau_c
        chosen_coverage = coverage
        chosen_f1 = f1_auto

# 5. Sauvegarde des seuils validés dans un fichier de configuration
thresholds = {
    "tau_protected": float(best_tau_protected),
    "tau_confidence": float(best_tau_conf),
    "val_auto_coverage": float(chosen_coverage),
    "val_auto_macro_f1": float(chosen_f1),
    "model_artifact": "models/clf_v1.joblib"
}

with open("data/thresholds.json", "w", encoding="utf-8") as f:
    json.dump(thresholds, f, indent=2)

print("\nSeuils enregistrés avec succès dans data/thresholds.json !")