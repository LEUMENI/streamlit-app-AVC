import streamlit as st
import numpy as np
from joblib import load

# Charger le modèle SVM
svm_model_best = load('svm_model.joblib')

# Fonction pour afficher le tableau de bord
def show_dashboard(avc_patients, no_avc_patients, show_avc, show_no_avc):
    st.sidebar.markdown("<h2 style='color: #007BFF;'>Dashboard</h2>", unsafe_allow_html=True)

    if st.sidebar.button("Liste des patients victimes d'AVC"):
        show_avc = True
    if st.sidebar.button("Liste des patients sans AVC"):
        show_no_avc = True

    if show_avc:
        st.sidebar.subheader("Patients avec AVC :")
        if avc_patients:
            for patient_id in avc_patients:
                st.sidebar.write(patient_id)
        else:
            st.sidebar.write("Aucun patient avec AVC")

    if show_no_avc:
        st.sidebar.subheader("Patients sans AVC :")
        if no_avc_patients:
            for patient_id in no_avc_patients:
                st.sidebar.write(patient_id)
        else:
            st.sidebar.write("Aucun patient sans AVC")

    return show_avc, show_no_avc

def main():
    st.set_page_config(page_title="Prédiction des AVC", page_icon=":brain:", layout="wide", initial_sidebar_state="expanded")

    # Utiliser st.session_state pour persister les données entre les reruns
    if 'avc_patients' not in st.session_state:
        st.session_state.avc_patients = []
    if 'no_avc_patients' not in st.session_state:
        st.session_state.no_avc_patients = []
    if 'show_avc' not in st.session_state:
        st.session_state.show_avc = False
    if 'show_no_avc' not in st.session_state:
        st.session_state.show_no_avc = False

    # CSS personnalisé
    st.markdown("""
        <style>
            body {
                background-color: #f0f8ff;
            }
            .stApp {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .stButton > button {
                background-color: #007BFF;
                color: #ffffff !important;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
                width: 100%;
                margin-top: 10px;
                margin-bottom: 10px;
                transition: background-color 0.3s, color 0.3s;
            }
            .stButton > button:focus {
                background-color: #0056b3;
                color: #ffffff !important;  /* Assurez-vous que la couleur reste blanche */
            }
            .stForm {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .stForm:hover {
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
            }
            .stTitle {
                color: #007BFF;
            }
            .stSidebar {
                background-color: #f8f9fa;
                padding: 20px;
                border-right: 1px solid #ddd;
            }
            .stMarkdown {
                color: #007BFF;
            }
        </style>
    """, unsafe_allow_html=True)

    # Section Formulaire avec titre et encadrement
    with st.form(key='avc_form'):
        st.title("Formulaire pour la Prédiction des Accidents Vasculaires Cérébraux")
        patient_id = st.text_input("Identifiant du patient", help="Entrez l'identifiant du patient (ex: IDM1)")
        age = st.number_input("Quel est l'âge du patient ?", min_value=0.0, step=1.0, help="Entrez l'âge en années.")
        hta_diabete = st.selectbox("Le patient souffre-t-il d'hypertension ou de diabète ?", ["Oui", "Non"], help="Sélectionnez Oui ou Non.")
        cardiopathie = st.selectbox("Le patient souffre-t-il d'une cardiopathie ?", ["Oui", "Non"], help="Sélectionnez Oui ou Non.")
        imc = st.number_input("Quelle est l'IMC du patient ?", min_value=0.0, step=0.01, help="Entrez l'Indice de Masse Corporelle.")
        sexe_encoded = st.selectbox("De quel sexe est le patient ?", ["Femme", "Homme"], help="Sélectionnez le sexe du patient.")
        tabac_alcool_encoded = st.selectbox("Le patient consomme-t-il du tabac ou de l'alcool ?", ["Non", "Oui"], help="Sélectionnez Oui ou Non.")
        submit_button = st.form_submit_button(label='Prédire')

    # Section Résultats de la Prédiction avec titre et encadrement
    if submit_button:
        st.title("Résultats de la Prédiction")
        if age > 0 and imc > 0 and hta_diabete and cardiopathie and sexe_encoded and tabac_alcool_encoded:
            patient_data = np.array([[age, hta_diabete == "Oui", cardiopathie == "Oui", imc, sexe_encoded == "Homme", tabac_alcool_encoded == "Oui", 1, 1]])
            prediction = svm_model_best.predict(patient_data)
            probability_avc = svm_model_best.predict_proba(patient_data)[0][1]

            if  probability_avc >= 0.5:
                st.success("Présence d'accident vasculaire cérébral chez ce patient.")
                st.session_state.avc_patients.append(patient_id)
            else:
                st.info("Absence d'accident vasculaire cérébral pour ce patient.")
                st.session_state.no_avc_patients.append(patient_id)

            st.markdown(f"<div class='stMarkdown'>La probabilité que ce patient soit atteint d'un accident vasculaire cérébral est de : {probability_avc:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stMarkdown'>Soit {probability_avc * 100:.2f} %</div>", unsafe_allow_html=True)
        else:
            st.error("Veuillez remplir tous les champs du formulaire.")

    # Afficher le tableau de bord dans la barre latérale
    st.session_state.show_avc, st.session_state.show_no_avc = show_dashboard(
        st.session_state.avc_patients, st.session_state.no_avc_patients,
        st.session_state.show_avc, st.session_state.show_no_avc
    )

if __name__ == '__main__':
    main()
