import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
import numpy as np
np.bool = bool
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

train_url = "https://github.com/Amincheminform/phKMOi_v1/raw/main/0_train_hKMOi.csv"
test_url = "https://github.com/Amincheminform/phKMOi_v1/raw/main/0_test_hKMOi.csv"

# https://github.com/Amincheminform/phKMOi_v1/blob/main/0_train_hKMOi.csv
# https://github.com/Amincheminform/phKMOi_v1/blob/main/0_test_hKMOi.csv

train_data = pd.read_csv(train_url, sep=',')
test_data = pd.read_csv(test_url, sep=',')
PandasTools.AddMoleculeColumnToFrame(train_data, 'Smiles', 'Molecule')
PandasTools.AddMoleculeColumnToFrame(test_data, 'Smiles', 'Molecule')

# Streamlit
logo_url = "https://raw.githubusercontent.com/Amincheminform/phKMOi_v1/main/phKMOi_v1_logo.jpg"

st.set_page_config(
    page_title="phKMOi_v1.0: predictor of hKMO inhibitor",
    layout="wide",
    page_icon=logo_url
)

st.sidebar.image(logo_url)
st.sidebar.success("Thank you for using phKMOi_v1!")

calc = Calculator(descriptors, ignore_3D=True)
descriptor_columns = ['MDEC-33', 'AXp-6d', 'BCUTd-1l', 'SMR_VSA7', 'Xch-7d', 'Xch-6d', 'AATS0d', 'AATS5p']

# Train the model
try:
    X_train, y_train = train_data[descriptor_columns], train_data['Binary']
    X_test, y_test = test_data[descriptor_columns], test_data['Binary']

    model = RandomForestClassifier(
        n_estimators=18, max_depth=7, min_samples_split=4,
        min_samples_leaf=1, random_state=42
    )
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    st.sidebar.success(f"Model trained with test accuracy: {test_accuracy:.2f}")

except Exception as e:
    st.sidebar.error(f"Model training failed: {e}")
    model = None

def generate_2d_image(smiles, img_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=img_size, kekulize=True) if mol else None

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())  # optionally customize
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

def get_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fp = Chem.RDKFingerprint(mol)
    return fp

st.title("phKMOi_v1.0: predictor of human KMO inhibitor(s)")

with st.expander("What is phKMOi_v1.0?", expanded=True):
    st.write('''*phKMOi_v1.0* is a tool to predict inhibitor(s) of human kynurenine 3-monooxygenase (KMO) enzyme. 
    Kynurenine 3 monooxygenase is a key enzyme in the kynurenine pathway (KP). hKMOi represent a novel therapeutic 
    approach with broad implications in the treatment of neurodegenerative disease, psychiatric disorders, 
    acute pancreatitis and immune mediated conditions.
    
    Example SMILES: CHDI-340246 (Known hKMOi): n1c(cc(nc1)C(=O)O)c1ccc(c(c1)Cl)OC1CC1
    
    ''')

smiles_input = st.text_input("Enter the SMILES string of a molecule:")

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        # Author : Dr. Sk. Abdul Amin
        # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

        all_data = pd.concat([train_data, test_data], ignore_index=True)

        query_fp = get_ecfp4(smiles_input)
        all_data['Fingerprint'] = all_data['Smiles'].apply(lambda x: get_ecfp4(x))
        all_data['Tanimoto'] = all_data['Fingerprint'].apply(lambda x: DataStructs.TanimotoSimilarity(query_fp, x))

        most_similar = all_data.loc[all_data['Tanimoto'].idxmax()]
        similar_smiles = most_similar['Smiles']
        similar_mol = most_similar['Molecule']

        most_active = all_data.loc[all_data['pIC50'].idxmax()]
        active_smiles = most_active['Smiles']
        active_mol = most_active['Molecule']

        st.subheader("Results")
        st.sidebar.success("Calculation may take < 30 seconds")
        st.sidebar.success("Thank you for your patience!")

        smiles_list = [smiles_input, similar_smiles, active_smiles]
        molecules = [Chem.MolFromSmiles(sm) for sm in smiles_list]

        descriptor_df = calc.pandas(molecules)
        #external_descriptor_df = descriptor_df[descriptor_columns].dropna()
        external_descriptor_df = descriptor_df[descriptor_columns].fillna(0)
        external_descriptor_df.replace([np.inf, -np.inf], 0, inplace=True)
        external_descriptor_df = external_descriptor_df.astype(float)
        
        X_external = external_descriptor_df
        X_external_np = X_external.to_numpy(dtype=np.float64)


        y_external_pred = model.predict(X_external_np)

        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_external)

        titles = [
            f"Query molecule\nPredicted Class: {y_external_pred[0]}",
            f"Most similar molecule from dataset\nTanimoto similarity: {most_similar['Tanimoto']:.2f}\nPredicted Class: {y_external_pred[1]}",
            f"Most active molecule from dataset\npIC50: {most_active['pIC50']:.2f}\nPredicted Class: {y_external_pred[2]}"
        ]

        col1, col2, col3 = st.columns(3)

        # Author : Dr. Sk. Abdul Amin
        # My [paper](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

        with col1:
            # SHAP plot for query molecule
            fig1, ax1 = plt.subplots(figsize=(7, 7))
            shap.plots.waterfall(shap_values[0, :, y_external_pred[0]], max_display=10, show=False)
            ax1.set_title(titles[0], fontsize=18, fontweight='bold')
            st.pyplot(fig1)
            mol_img = mol_to_array(molecules[0])
            st.image(mol_img, caption="Query Molecule", width=250)
            #st.markdown("**Query Molecule**")
            #st.image(mol_img, width=250)

        with col2:
            # SHAP plot for most similar molecule
            fig2, ax2 = plt.subplots(figsize=(7, 7))
            shap.plots.waterfall(shap_values[1, :, y_external_pred[1]], max_display=10, show=False)
            ax2.set_title(titles[1], fontsize=18, fontweight='bold')
            st.pyplot(fig2)
            similar_mol_img = mol_to_array(molecules[1])
            st.image(similar_mol_img, caption=f"Most similar molecule\nTanimoto similarity: {most_similar['Tanimoto']:.2f}", width=250)
            #st.markdown("**Most similar molecule\nTanimoto similarity**: {**most_similar**[**'Tanimoto'**]:.2f}")
            #st.image(similar_mol_img, width=250)

        with col3:
            # SHAP plot for most active molecule
            fig3, ax3 = plt.subplots(figsize=(7, 7))
            shap.plots.waterfall(shap_values[2, :, y_external_pred[2]], max_display=10, show=False)
            ax3.set_title(titles[2], fontsize=18, fontweight='bold')
            st.pyplot(fig3)
            active_mol_img = mol_to_array(molecules[2])
            st.image(active_mol_img, caption="Most active molecule", width=250)
            #st.markdown("**Most active molecule**")
            #st.image(active_mol_img, width=250)

else:
    st.info("Please enter a SMILES string to get predictions.")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/Amincheminform)

        #### Contact Us
        - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
    ''')
