import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Configure the page
st.set_page_config(
    page_title="Iris SDataset",
    layout="wide",
)
st.header(":blue[Visualisez la Base de sonnée Iris ]", divider="gray")
with st.sidebar:
    st.html("""
    <div>
        <h1> Wesley Eliel MONTCHO </h1>
        <h2> Software Developer </h2>
        <ul style='padding-left:40px;'>
            <li>
                <span>Email:</span>
                <span style='text-decoration: underline gray;'>wesleymontcho@gmail.com</span>
             </li>
            <li>
                <span>WhatsApp:</span>
                <span>+229 01 69196638</span></li>
            <li>
                <span>LinkedIn:</span>
                <span ><a style='text-decoration: underline gray;'>Wesley E. MONTCHO</a> </span></li>
        </ul>
    </div>
    """
            )

with st.expander("Importation de données", True):
    _data_source_options = {
        'Importer les données': 'import',
        "Utiliser l' existant": 'existing',
    }

    # Create radio buttons using custom labels and slugs
    _data_source = st.radio("Veuillez :", [option for option in _data_source_options.keys()])

    st.text(_data_source)
    # Display selected slug
    if _data_source_options[_data_source] == "import":
        file = st.file_uploader("Choisissez le fichier", type=["csv", "xlsx", "xls"])
    else:
        file = "iris.csv"
    if file:
        dataframe = pd.read_csv(file)
        len_file = dataframe.shape[0]
        number_of_lines_to_display = st.slider("Nombre de lignes à traiter ?", 0, len_file, len_file)

if file:
    print("\n\n\n\n\n")
    print(file)
    print("\n\n\n\n\n")
    dataframe = pd.read_csv(file)
    print("\n\n\n\n\n")
    print(dataframe)
    print("\n\n\n\n\n")
    iris = dataframe.iloc[:number_of_lines_to_display]
    with st.expander("Visualisation des données", True):
        # The actual dataset info
        st.subheader("Les informations générales")
        st.write(iris.info())

        # Shapes
        st.subheader("Les dimensions")
        st.write(f"Dataset shape: {iris.shape}")

        # Is the missing values
        st.subheader("Les valeurs manquantes")
        st.write(iris.isnull().sum())

        # Write dats types
        st.subheader("Les types de données")
        st.write(iris.dtypes)

        # Write stats
        st.subheader("Les statistiques globales")
        st.write(iris.describe())

        st.subheader("Les données")
        st.write(iris)

    # Histogram
    with st.expander("Histogrammes des données", True):
        st.subheader("Histogrammes")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(iris['sepal_length'], bins=20)
        ax2.hist(iris['petal_length'], bins=20)
        st.pyplot(fig)

    with st.expander("Encoded Species", True):
        # Encode categorical variables:
        le = LabelEncoder()
        iris['species_encoded'] = le.fit_transform(iris['species'])
        st.subheader("Encodages des valeurs non discrètes ( Species )")
        st.write(iris[['species', 'species_encoded']])

    # Create a scatter plot of sepal length vs petal length:
    with st.expander("Tailles des Sepals vs Tailles des Petal Length", True):
        st.subheader("Tailles des Sepal vs Tailles des Petal")
        plt.scatter(iris['sepal_length'], iris['petal_length'], c=iris['species_encoded'])
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        st.pyplot(plt.gcf())

    copied_iris = iris.drop("species", axis=1)

    # Display the correlation matrix:
    with st.expander("Matrice de correlation", True):
        st.write(copied_iris.corr(numeric_only=True))

    # Normalisation errors
    # Data Transformation Standardize numerical features:
    with st.expander("Normalisation des données", True):
        scaler = StandardScaler()
        iris_scaled = pd.DataFrame(scaler.fit_transform(copied_iris),
                                   columns=copied_iris.select_dtypes(include=['int64', 'float64']).columns)
        st.subheader("Données Normalisées")
        st.write(iris_scaled)

    # Machine Learning Train a simple classifier:
    with st.expander("Implementation d' un Modèle de Machine Learning", True):
        X = copied_iris[['sepal_length', 'petal_length']]
        y = copied_iris['species_encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        st.subheader("Performances du model")
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        st.write(f"Précision: {accuracy:.2f}")

    # Model testing:
    with st.expander("Test du model avec des valeurs personnelles", True):
        pass
