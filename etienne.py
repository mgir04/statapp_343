#implémentation de la méthode ElasticNet

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

#Exemples en attendant d'avoir les données
#Dans X on met le tableau omique
#Dans Y on met des scores de prédiction cardiaques (il faut déjà calculer ça) -> dans el tableau?
n_samples, n_features = 335, 17000
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

#Normalisation pour que les pénalisations sur les variables soient cohérentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#On lance l'entraînement pour plusierus valeurs : 
#On se rapproche du modèle de Ridge quand on tend vers 0 et du modèle Lasso quand on tend vers 1
#Je teste bcp de valeurs proches de 1 car vu le nombre de colonnes on va probablement chercher à éliminer du monde

elastic_net_model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], 
    cv=5, 
    random_state=42,
    max_iter=10000
)


elastic_net_model.fit(X_scaled, y)


coefficients = elastic_net_model.coef_

#on regarde ce que le meilleru modèle a gardé.

selected_indices = np.where(coefficients != 0)[0]
print(f"Nombre de gènes/métabolites sélectionnés par le modèle : {len(selected_indices)}")
print(f"Alpha optimal choisi par validation croisée : {elastic_net_model.alpha_}")