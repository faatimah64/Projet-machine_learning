import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données avec cache
@st.cache_data
def charger_donnees():
    return pd.read_csv("Fuel_Consumption.csv")

# Fonction pour afficher la moyenne des émissions de CO2 par rapport au nombre de cylindres
def afficher_moyenne_co2_par_cylindre(data):
    st.subheader("Moyenne des émissions de CO2 par rapport au nombre de cylindres")

    moyenne_co2_par_cylindres = data.groupby('CYLINDERS')['CO2EMISSIONS'].mean()
    st.bar_chart(moyenne_co2_par_cylindres, use_container_width=True)

# Fonction pour afficher la matrice de corrélation
def afficher_matrice_correlation(data):
    st.subheader("Matrice de corrélation")

    matrice_correlation = data.corr()
    st.write(matrice_correlation)

# Fonction pour entraîner et évaluer le modèle de régression linéaire
def entrainer_et_evaluer_modele(data, x, y, nb_lignes_prediction):
    st.title("Régression des émissions de CO2")

    x_entrainement, x_test, y_entrainement, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    st.write("Dimension des données d'entraînement : ", x_entrainement.shape, y_entrainement.shape)
    st.write("Dimension des données de validation : ", x_validation.shape, y_validation.shape)
    st.write("Dimension des données de test : ", x_test.shape, y_test.shape)

    modele_regression_lineaire = LinearRegression()
    modele_regression_lineaire.fit(x_entrainement, y_entrainement)

    # Prédiction pour les premières lignes par défaut
    nb_lignes_prediction = min(nb_lignes_prediction, len(x_test))
    x_prediction = x_test[:nb_lignes_prediction]
    y_prediction = modele_regression_lineaire.predict(x_prediction)

    # Affichage des prédictions
    st.subheader(f"Prédiction pour les {nb_lignes_prediction} premières lignes de test :")
    for i in range(nb_lignes_prediction):
        st.write(f"Ligne {i+1}: Prédiction = {y_prediction[i][0]:.2f} | Valeur réelle = {y_test[i][0]:.2f}")


# Interface utilisateur Streamlit
st.title("Tableau de Bord Fuel Consumption")

donnees = charger_donnees()

page = st.sidebar.radio("Sélectionnez la section :", ["Aperçu", "Moyenne CO2 par cylindre", "Matrice de corrélation", "Régression des émissions de CO2"])

if page == "Aperçu":
    st.subheader("Aperçu du jeu de données")
    st.write(donnees.head())
elif page == "Moyenne CO2 par cylindre":
    afficher_moyenne_co2_par_cylindre(donnees)
elif page == "Matrice de corrélation":
    afficher_matrice_correlation(donnees)
elif page == "Régression des émissions de CO2":
    caracteristiques_selectionnees = st.sidebar.multiselect("Sélectionnez les caractéristiques pour la régression", donnees.columns[:-1])
    nb_lignes_prediction = st.sidebar.slider("Nombre de lignes pour la prédiction", min_value=1, max_value=20, value=5)
    
    if caracteristiques_selectionnees:
        x = donnees[caracteristiques_selectionnees].values
        y = donnees['CO2EMISSIONS'].values.reshape(-1, 1)
        entrainer_et_evaluer_modele(donnees, x, y, nb_lignes_prediction)

st.sidebar.text("Mon App")
