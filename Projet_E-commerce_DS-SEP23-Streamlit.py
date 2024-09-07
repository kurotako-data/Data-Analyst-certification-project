# LIBRARY IMPORT

import streamlit as st

# Set site to wide Width
st.set_page_config(layout = "wide")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import warnings
import base64
from pathlib import Path

# MACHINE LEARNING
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

# Autres modeles
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM

#from imblearn.metrics import geometric_mean_score

#CSS directement intégré
st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        p {
            font-size: 18px;
            line-height: 1.6;
        }
        /* Ajoutez d'autres styles ici selon vos besoins */
        .custom-class {
            background-color: #e0e0e0;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)


####################################################################################################
# PAGE BEGIN CACHE FUNCTIONS
####################################################################################################

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html


warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# Cached function to load csv
@st.cache_data
def load_data(url):
  df = pd.read_csv(url)
  return df

# Import CSV
df_it_prop1 = load_data("./item_properties_part1 copie.csv")
df_it_prop2 = load_data("./item_properties_part2 copie.csv" )
df = load_data("./events copie.csv")
category_tree = load_data("./category_tree copie.csv" )

@st.cache_data
def create_item(df1, df2):
  # Concaténation des 2 fichiers qui ont la même structure en un df nommée item
  item = pd.concat([df1, df2], axis = 0)
  return item

item = create_item(df_it_prop1, df_it_prop2)

@st.cache_data
def create_liste_itemid_item(item):
  # Nbre d'itemid dans item
  liste_itemid_item = item.itemid.unique()
  return liste_itemid_item

liste_itemid_item = create_liste_itemid_item(item)

@st.cache_data
def transform_step1(df, liste_itemid_item):
  # Filtrage de df avec les itemid commun à item et df
  df = df.loc[df['itemid'].isin(liste_itemid_item)]
  df = df.reset_index(drop = True)
  return df

df = transform_step1(df, liste_itemid_item)

# Seules les informations sur la disponibilité ou non du produit ainsi que sa catégorie peuvent être utiles dans item

@st.cache_data
def create_item_availability(item):
  # Création de 2 df item_ availability et item_categ
  item_availability = item.loc[item.property == 'available']
  item_availability = item_availability.reset_index(drop = True)
  item_availability.itemid = item_availability.itemid.astype('int64')
  return item_availability

item_availability = create_item_availability(item)

@st.cache_data
def create_item_category(item):
  item_category = item.loc[item.property == 'categoryid']
  item_category = item_category.reset_index(drop = True)
  item_category.itemid = item_category.itemid.astype('int64')
  return item_category

item_category = create_item_category(item)

@st.cache_data
def create_merged_1(df, item_availability):
  # Fusion de df avec item_availability pour récupérer les infos sur la disponibilité de nos produits  
  merged_1 = pd.merge_asof(df.sort_values('timestamp'),item_availability.sort_values('timestamp'),by = 'itemid', on = 'timestamp',direction = 'nearest')
  return merged_1

merged_1 = create_merged_1(df, item_availability)

@st.cache_data
def create_merged_2(merged_1, item_category):
  # Création merged_2 pour récupérer à présent les categories de certains produits

  merged_2 = pd.merge_asof(merged_1.sort_values('timestamp'),item_category.sort_values('timestamp'),by = 'itemid', on = 'timestamp',direction = 'nearest')
  # Le df nommé merged_2 comporte ainsi toutes les informations récupérées du df item et du df(event)
  # Néttoyage du DataFrame merged_2 :

  # Sauvegarde merged_2 avant le drop
  merged_2_tmp = merged_2

  # Suppression des colonnes property et transactionid
  merged_2 = merged_2.drop(['property_x','property_y','transactionid'], axis=1)

  # Renommage des colonnes pour plus de clarté
  merged_2 = merged_2.rename(columns = {'value_x': "available", "value_y": "categoryid"})

  merged_2.categoryid = merged_2.categoryid.astype(int)
  return merged_2, merged_2_tmp

merged_2, merged_2_tmp = create_merged_2(merged_1, item_category)

@st.cache_data
def create_df_final_step1(merged_2):
  # Dernière fusion du df merged_2 avec ce 4ème fichier pour récupérer les parentid
  # Nous allons à présent récuperer les informations sur les parentid correspondant à nos catégories dans category_tree
  df_final = merged_2.merge(category_tree, how = 'left', on = 'categoryid')

  # Création de 3 nouvelles variables à partir de la colonne event
  df_final = df_final.join(pd.get_dummies(data = df_final['event']))

  # Conversion en datetime de timestamp
  df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], unit = 'ms')

  # Créer la Colonne Mois
  df_final['month'] = df_final['timestamp'].dt.month

  # Remplacement des fillna de parentid par 9999 qui correspond à Other
  df_final['parentid'].fillna(9999, inplace = True)

  # Conversion des variables
  df_final[['available','parentid','addtocart','transaction','view']] = df_final[['available','parentid','addtocart','transaction','view']].astype(int)
  return df_final

df_final = create_df_final_step1(merged_2)

@st.cache_data
def create_df_final_step2(df_final):
  # Aperçu des doublons
  duplicates_df_final = df_final[df_final.duplicated(keep = False)]
  return duplicates_df_final

duplicates_df_final = create_df_final_step2(df_final)

@st.cache_data
def create_df_final_step3(df_final):
  # Suppression doublons car ils sont identiques
  df_final = df_final.drop_duplicates()
  df_final = df_final.drop(['event','timestamp'], axis = 1)
  return df_final

df_final = create_df_final_step3(df_final)

@st.cache_data
def create_df_final_ag(df_final):
  # Agrégation des données grâce au groupby sur les variables, application des fonctions d'agrégation différentes en fonction des variables :

  # addtocart   : somme
  # transaction : somme
  # view        : somme
  # available   : garder la dernière valeur
  # categoryid  : garder la première valeur
  # parentid    : garder la première valeur
  # month       : garder le nombre de mois unique
  # visitorid   : garder le nombre de visiteurs uniques qui ont consulté le produit

  # Agregation des données
  dictag = {'addtocart':'sum','transaction':'sum','view':'sum','available':'last','categoryid':'first','parentid':'first','month':'nunique','visitorid':'nunique' }
  df_final_ag = df_final.groupby('itemid').agg(dictag)
  return df_final_ag

df_final_ag = create_df_final_ag(df_final)

####################################################################################################
# PAGE END CACHE FUNCTIONS
####################################################################################################

st.sidebar.title("Sommaire")
pages=["Accueil", "I- Exploration des données", "II-Transformation et Pré-processing", "III- Datavisualisation", "IV- Machine learning" , "V- Interprétation et conclusion"]
page=st.sidebar.radio("Aller vers", pages)

###########################################################################################################################################################################################################
# Accueil
###########################################################################################################################################################################################################
if page == pages[0] :
  
  st.title("Projet E-commerce DS - SEP23")
  st.markdown("<h4 class = 'bordered_blue'>Projet E-commerce: Comprendre le comportement des utilisateurs du site et prédire leurs comportements futurs</h4>", unsafe_allow_html = True)
  st.write("")  
  st.image('./images/data-analyst.png', width = 2700)
  st.markdown("<h4 class = 'bordered'>Membres du porjet</h4>", unsafe_allow_html = True)
  st.write("")
  col1, col2, col3, col4, col5 = st.columns(5)
  with col1:
    st.markdown("<h6 class = 'centered'>Ahmad Benomari</h6>", unsafe_allow_html = True)
    st.markdown("<p style='text-align: center;'>"+ img_to_html('./images/ahmad_benomari.jpg') + "</p>", unsafe_allow_html = True)
  with col2:
    st.markdown("<h6 class = 'centered'>Marc Benedetto</h6>", unsafe_allow_html = True)
    st.markdown("<p style='text-align: center;'>"+ img_to_html('./images/marc_benedetto.jpg') + "</p>", unsafe_allow_html = True)
  with col3:
    st.markdown("<h6 class = 'centered'>Michèle Gaba</h6>", unsafe_allow_html = True)
    st.markdown("<p style='text-align: center;'>"+ img_to_html('./images/michele_gaba.jpg') + "</p>", unsafe_allow_html = True)
  with col4:
    st.markdown("<h6 class = 'centered'>Partice Giardino</h6>", unsafe_allow_html = True)
    st.markdown("<p style='text-align: center;'>"+ img_to_html('./images/partice_giardino.png') + "</p>", unsafe_allow_html = True)
  with col5:
    st.markdown("<h6 class = 'centered'>Radouan Chemmaa</h6>", unsafe_allow_html = True)
    st.markdown("<p style='text-align: center;'>"+ img_to_html('./images/radouan_chemmaa.jpg') + "</p>", unsafe_allow_html = True)

###########################################################################################################################################################################################################
# Exploration des datasets
###########################################################################################################################################################################################################
if page == pages[1] : 

  st.markdown("<h4 class = 'bordered_blue'>I - Exploration des données</h4>", unsafe_allow_html = True)
  st.write("")
  st.markdown("<h5 class = 'bordered'>Importation et affichage informations de chaque DataFrame</h5>", unsafe_allow_html = True)
  st.write("")
  
  st.markdown("<h5 class = 'underlined'>Fichier 1: item_properties_part1 : 2520260 lignes, 4 colonnes</h5>", unsafe_allow_html = True)
  col1, col2 = st.columns(2)
  with col1:
    st.write("Aperçu du DataFrame")
    st.dataframe(df_it_prop1.head(5))
  with col2:
    st.write("Infos du DataFrame")
    buffer = io.StringIO()
    df_it_prop1.info(buf = buffer)
    s = buffer.getvalue()
    st.text(s)
  st.image('./images/descr/detail_set_item_properties.png', width = 1440)

  st.markdown("""---""")

  st.markdown("<h5 class = 'underlined'>Fichier 2: item_properties_part2 : 2 115 992 lignes, 4 colonnes</h5>", unsafe_allow_html = True)
  col1, col2 = st.columns(2)
  with col1:
    st.write("Aperçu du DataFrame")
    st.dataframe(df_it_prop2.head(5))
  with col2:
    st.write("Infos du DataFrame")
    buffer = io.StringIO()
    df_it_prop2.info(buf = buffer)
    s = buffer.getvalue()
    st.text(s)
  st.image('./images/descr/detail_set_item_properties.png', width = 1440)

  st.markdown("""---""")

  st.markdown("<h5 class = 'underlined'>Fichier 3: events : 275 610 lignes, 5 colonnes</h5>", unsafe_allow_html = True)
  col1, col2 = st.columns(2)
  with col1:
    st.write("Aperçu du DataFrame")
    st.dataframe(df.head(5))
  with col2:
    st.write("Infos du DataFrame")
    buffer = io.StringIO()
    df.info(buf = buffer)
    s = buffer.getvalue()
    st.text(s)
  st.image('./images/descr/detail_set_events.png', width = 1440)

  st.markdown("""---""")

  st.markdown("<h5 class = 'underlined'>Fichier 4: category_tree : 1669 lignes, 2 colonnes</h5>", unsafe_allow_html = True)
  col1, col2 = st.columns(2)
  with col1:
    st.write("Aperçu du DataFrame")
    st.dataframe(category_tree.head(5))
  with col2:
    st.write("Infos du DataFrame")
    buffer = io.StringIO()
    category_tree.info(buf = buffer)
    s = buffer.getvalue()
    st.text(s)
  st.image('./images/descr/detail_set_category_tree.png', width = 1440)

###########################################################################################################################################################################################################
# Transformation et Préprocessing
###########################################################################################################################################################################################################
if page == pages[2] :
  st.markdown("<h4 class = 'bordered_blue'>II - Transformation et Pré-processing</h4>", unsafe_allow_html = True)
  st.write("")
  
  def tranform_and_process(methode):
      if methode == 'Choix du détail':
        st.write("")
        st.markdown("<h5 class = 'underlined'>1. Création des DataFrame transformés df > merge_1 > merge_2</h5>", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	DataFrame le plus pertinent :</span> 'events' (df), récupération des informations essentielles du DataFrame 'item' (item) et intégration au DataFrame 'event'.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Récupération des items communs aux deux DataFrame :</span> Parcours des items uniques et commun puis stockage dans une liste nommée 'liste_itemid_item'.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Filtrage sur les 'property' pertinentes :</span> Seule la disponibilité et la catégorie des produits est pertinente étant donné que le reste des informations est anonymisé.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Ajout disponibilité et catégorie :</span> Fusion de df avec item_availability via pd.merge_asof pour ajouter la disponibilité dans le DataFrame merged_1 puis fusion de merged_1 avec item_category via pd.merge_asof" + 
                    "pour ajouter la catégorie dans le DataFrame merged_2.", unsafe_allow_html = True)
        st.markdown("""---""")
        st.markdown("<h5 class = 'underlined'>2. Nettoyage du DataFrame merged_2</h5>", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Suppression de colonnes :</span> Suppression de la colonne 'property' (remplacée par availability et item_category) et 'transactionid' qui n'a aucune valeur ajoutée.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Changement nom de colonnes :</span> Renommage des colonnes pour plus de clarté.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Vérification des types de données :</span> Examen des types de données pour s'assurer qu'ils sont adaptés aux analyses prévues.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Conversion des types de données :</span> Conversion des données dans le format requis.", unsafe_allow_html = True)
        st.markdown("""---""")
        st.markdown("<h5 class = 'underlined'>3. Création du DataFrame df_final</h5>", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Fusion des DataFrame :</span> Fusion de merged_2 avec category_tree dans le DataFrame df_final.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Conversion des types de données :</span> Conversion des données dans le format requis.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Ajout Mois :</span> Création d'une colonne month à partir de la colonne timestamp.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Identification des NaNs :</span> Seule la colonne 'parentid' contient des valeurs manquantes.", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Remplacement des NaNs :</span> Remplacement des NaNs par la valeur spécifique 9999 (catégorie non utilisée)", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Conversion des types de données :</span> Conversion des données dans le format requis.", unsafe_allow_html = True)
        st.markdown("""---""")
        st.markdown("<h5 class = 'underlined'>4. Agrégation des données du DataFrame df_final</h5>", unsafe_allow_html = True)
        st.markdown("<span class='bold_margin'>o	Groupement par clés spécifiques :</span> Regroupement des données par une ou plusieurs lignes.", unsafe_allow_html = True)
        st.write("addtocart   : somme")
        st.write("transaction : somme")
        st.write("view        : somme")
        st.write("available   : garder la dernière valeur")
        st.write("categoryid  : garder la première valeur")
        st.write("parentid    : garder la première valeur")
        st.write("month       : garder le nombre de mois unique")
        st.write("visitorid   : garder le nombre de visiteurs uniques qui ont consulté le produit")
        st.markdown("""---""")
        st.markdown("<h5 class = 'underlined'>5. Aperçu du DataFrame df_final_ag</h5>", unsafe_allow_html = True)
        st.write("Aperçu du DataFrame")
        st.write(df_final_ag.head())
        st.write("Shape du DataFrame")
        st.write(df_final_ag.shape)
        st.write("Description du DataFrame")
        st.write(df_final_ag.describe())

      elif methode == '1. Création des DataFrame transformés df > merge_1 > merge_2':
        st.markdown("<h5 class = 'bordered_green'>1. Création du DataFrame merged_1</h5>", unsafe_allow_html = True)
        st.write("")
        st.write("Le DataFrame le plus pertinent pour répondre au projet est : 'events' (df).")
        st.markdown("<span class = 'underlined'>Récupération des informations essentielles du DataFrame 'item' (item) et intégration au DataFrame 'event' > item = create_item(df_it_prop1, df_it_prop2)</span>", unsafe_allow_html = True)
        st.write(item.head(10))
        st.markdown("<span class = 'underlined'>Parcours des items uniques et commun puis stockage dans une liste nommée 'liste_itemid_item' :</span>", unsafe_allow_html = True)
        st.write("Parcours des items uniques et commun puis stockage dans une liste nommée 'liste_itemid_item' :")
        st.markdown("<span class = 'margin_code'>liste_itemid_item = create_liste_itemid_item(item)</span>", unsafe_allow_html = True)
        st.write(liste_itemid_item)
        st.write("Mise à jour de df en ne gardant que les items uniques et communs :")
        st.markdown("<span class = 'margin_code'>df = df.loc[df['itemid'].isin(liste_itemid_item)]</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df = df.reset_index(drop = True)</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'underlined'>Seule la disponibilité et la catégorie des produits est pertinente étant donné que le reste des informations est anonymisé.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>item_availability = create_item_availability(item)</span>", unsafe_allow_html = True)
        st.write(item_availability.head(10))
        st.markdown("<span class = 'margin_code'>item_category = create_item_category(item)</span>", unsafe_allow_html = True)
        st.write(item_category.head(10))
        st.markdown("<span class = 'underlined'>Ajout disponibilité et la catégorie</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>merged_1 = pd.merge_asof(df.sort_values('timestamp'),item_availability.sort_values('timestamp'),by ='itemid', on ='timestamp',direction ='nearest')</span>", unsafe_allow_html = True)
        st.write(merged_1.head(10))
        st.markdown("<span class = 'margin_code'>merged_2 = pd.merge_asof(merged_1.sort_values('timestamp'),item_category.sort_values('timestamp'),by ='itemid', on ='timestamp',direction ='nearest')</span>", unsafe_allow_html = True)
        st.write(merged_2_tmp.head(10))

      elif methode == '2. Nettoyage du DataFrame merged_2':
        st.markdown("<h5 class = 'bordered_green'>2. Nettoyage du DataFrame merged_2</h5>", unsafe_allow_html = True)
        st.write("")
        st.markdown("<span class = 'underlined'>Suppression de la colonne 'property' (remplacée par availability et item_category) et 'transactionid' qui n'a aucune valeur ajoutée.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'underlined'>Renommage des colonnes pour plus de clarté.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>merged_2 = merged_2.rename(columns = {'value_x': 'available', 'value_y': 'categoryid'})</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'underlined'>Examen des types de données pour s'assurer qu'ils sont adaptés aux analyses prévues.</span>", unsafe_allow_html = True)
        st.write(merged_2_tmp.dtypes)
        st.markdown("<span class = 'underlined'>Conversion des données dans le format requis.</span>", unsafe_allow_html = True)
        st.write(merged_2.dtypes)

      elif methode == '3. Création du DataFrame df_final':
        st.markdown("<h5 class = 'bordered_green'>3. Création du DataFrame df_final</h5>", unsafe_allow_html = True)
        st.write("")
        st.markdown("<span class = 'underlined'>Fusion de merged_2 avec category_tree dans le DataFrame df_final.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final = merged_2.merge(category_tree, how = 'left', on = 'categoryid')</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final = df_final.join(pd.get_dummies(data = df_final['event']))</span>", unsafe_allow_html = True)

        st.markdown("<span class = 'underlined'>Conversion des données dans le format requis.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], unit = 'ms')</span>", unsafe_allow_html = True)

        st.markdown("<span class = 'underlined'>Création d'une colonne month à partir de la colonne timestamp.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final['month'] = df_final['timestamp'].dt.month</span>", unsafe_allow_html = True)

        st.markdown("<span class = 'underlined'>Identification des NaNs : Seule la colonne 'parentid' contient des valeurs manquantes.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>duplicates_df_final = df_final[df_final.duplicated(keep = False)]</span>", unsafe_allow_html = True)

        st.write(duplicates_df_final.dtypes)
        st.markdown("<span class = 'underlined'>Remplacement des NaNs par la valeur spécifique 9999 (catégorie non utilisée)</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final['parentid'].fillna(9999, inplace = True)</span>", unsafe_allow_html = True)

        st.markdown("<span class = 'underlined'>Conversion des données dans le format requis.</span>", unsafe_allow_html = True)
        st.markdown("<span class = 'margin_code'>df_final[['available','parentid','addtocart','transaction','view']] = df_final[['available','parentid','addtocart','transaction','view']].astype(int)</span>", unsafe_allow_html = True)
        st.write(df_final.dtypes)

      elif methode == '4. Agrégation des données du DataFrame df_final':
        st.markdown("<h5 class = 'bordered_green'>4. Agrégation des données du DataFrame df_final</h5>", unsafe_allow_html = True)
        st.write("")
        st.write("Agrégation des données grâce au groupby sur les variables, application des fonctions d'agrégation différentes en fonction des variables")
        
        st.write("addtocart   : somme")
        st.write("transaction : somme")
        st.write("view        : somme")
        st.write("available   : garder la dernière valeur")
        st.write("categoryid  : garder la première valeur")
        st.write("parentid    : garder la première valeur")
        st.write("month       : garder le nombre de mois unique")
        st.write("visitorid   : garder le nombre de visiteurs uniques qui ont consulté le produit")

      elif methode == '5. Aperçu du DataFrame df_final_ag':
        st.markdown("<h5 class = 'bordered_green'>5. Aperçu du DataFrame df_final</h5>", unsafe_allow_html = True)
        st.write("")
        st.markdown("<h5>Aperçu du df_final_ag qui servira pour les parties Visualisation et Modèles de Machine Learning</h5>", unsafe_allow_html = True)
    
        st.write("Aperçu du DataFrame")
        st.write(df_final_ag.head())
        st.write("Shape du DataFrame")
        st.write(df_final_ag.shape)
        st.write("Description du DataFrame")
        st.write(df_final_ag.describe())

  choix = ['Choix du détail',
          '1. Création des DataFrame transformés df > merge_1 > merge_2',
          '2. Nettoyage du DataFrame merged_2',
          '3. Création du DataFrame df_final',
          '4. Agrégation des données du DataFrame df_final',
          '5. Aperçu du DataFrame df_final_ag']
  option = st.selectbox('Transformation / Pre-processing', choix)

  tranform_and_process(option)
  

###########################################################################################################################################################################################################
# Datavisualisation
###########################################################################################################################################################################################################
if page == pages[3] : 

  ####################################################################################################
  # PAGE BEGIN CACHE FUNCTIONS
  ####################################################################################################
  @st.cache_data
  def month_value_count_create(df_final, df_final_ag):

    month_value_count_dff = df_final.month.value_counts()
    month_value_count_dffag = df_final_ag.month.value_counts()

    return month_value_count_dff, month_value_count_dffag

  month_value_count_dff, month_value_count_dffag = month_value_count_create(df_final, df_final_ag)

  @st.cache_data
  def month_labels_create(df_to_convert):

    months_dic = {'1' : 'Janvier', '2' : 'Février', '3' : 'Mars', '4' : 'Avril', '5' : 'Mai', '6' : 'Juin', '7' : 'Juillet', '8' : 'Août', '9' : 'Septembre', '10' : 'Octobre', '11' : 'Novembre', '12' : 'Décembre'}
    labels = []

    for x, y in df_to_convert.items():
        for key, value in months_dic.items():
          if int(key) == x:
            labels.append(value)
    return labels

  month_value_count_dff_labels = month_labels_create(month_value_count_dff)

  @st.cache_data
  def product_availability_values_create(df_final_ag):
    product_availability_values = df_final_ag.available.astype(str).value_counts()
    return product_availability_values
  
  product_availability_values = product_availability_values_create(df_final_ag)

  @st.cache_data
  def product_values_create(df_final_ag):
    product_values_topview = df_final_ag.sort_values(by = 'view', ascending = False).head(10)
    product_values_topadd = df_final_ag.sort_values(by = 'addtocart', ascending = False).head(10)
    product_values_toptrans = df_final_ag.sort_values(by = 'transaction', ascending = False).head(10)

    product_values_topview.categoryid = product_values_topview.categoryid.astype('str')
    product_values_topadd.categoryid = product_values_topadd.categoryid.astype('str')
    product_values_toptrans.categoryid = product_values_toptrans.categoryid.astype('str')
    return product_values_topview, product_values_topadd, product_values_toptrans
  
  product_values_topview, product_values_topadd, product_values_toptrans = product_values_create(df_final_ag)  

  @st.cache_data
  def availability_labels_create(df_to_convert):

    label_dic = {'1' : 'Disponible', '0' : 'Non disponible'}
    labels = []

    for x, y in df_to_convert.items():
        for key, value in label_dic.items():
          if key == x:
            labels.append(value)
    return labels

  availability_labels = availability_labels_create(product_availability_values)

  @st.cache_data
  def related_view_add_trans01():
    fig3 = make_subplots(rows = 1, cols = 3,
                    subplot_titles = ['Relation entre : Vues et ajouts au panier','Relation entre : Ajouts au panier et transactions','Relation entre : Visiteurs et vues'])

    fig3.add_trace(go.Scatter(x     = df_final_ag.view,
                              y     = df_final_ag.addtocart,
                              marker_line = dict(width = 1, color = 'white'),
                              mode  = "markers"),
                              1, 1)

    fig3.add_trace(go.Scatter(x     = df_final_ag.addtocart,
                              y     = df_final_ag.transaction,
                              mode  = "markers"),
                              1, 2)
    fig3.add_trace(go.Scatter(x     = df_final_ag.visitorid,
                              y     = df_final_ag.view,
                              mode  = "markers"),
                              1, 3)
    
    # Forcer l'affichage des axes en blanc
    fig3.update_xaxes(showline = True, linewidth = 1, linecolor = 'white')
    fig3.update_yaxes(showline = True, linewidth = 1, linecolor = 'white')
    # Ajout legend sur les axes x et y
    fig3.update_xaxes(color = "white")
    fig3.update_xaxes(title_text = "Vues", row = 1, col = 1)
    fig3.update_xaxes(title_text = "Ajout au panier", row = 1, col = 2)
    fig3.update_xaxes(title_text = "Visiteur", row = 1, col = 3)

    fig3.update_yaxes(title_text = "Ajout au panier", row = 1, col = 1)
    fig3.update_yaxes(title_text = "Transaction", row = 1, col = 2)
    fig3.update_yaxes(title_text = "Vue", row = 1, col = 3)

    st.plotly_chart(fig3, use_container_width = True)

  @st.cache_data
  def related_view_add_visit01():

    col1, col2, col3 = st.columns(3)

    sns.set(rc={'axes.facecolor':'#0E1117', 'figure.facecolor':'#0E1117', 'axes.grid': False, 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
    with col1:
      ax = sns.lmplot(x ='visitorid', y ="view", data = df_final_ag, order = 2, line_kws = {'color': 'red'}, height = 5, aspect = 1)
      plt.title("Relation visiteurs uniques et nombre de vues", color = 'white', fontsize = 8)
      plt.xlabel('ID visiteur', fontsize = 6)
      plt.ylabel('Vues', fontsize = 6)
      # Taille des labels
      plt.xticks(fontsize = 5)
      plt.yticks(fontsize = 5) 
      st.pyplot(plt)
    with col2:
      ax = sns.lmplot(x ='visitorid', y ="addtocart", data = df_final_ag, order = 2, line_kws = {'color': 'red'}, height = 5, aspect = 1)
      plt.title("Relation nombre de visiteurs uniques et nombre d'ajouts au panier", color = 'white', fontsize = 8)
      plt.xlabel('ID visiteur', fontsize = 6)
      plt.ylabel('Ajouts au panier', fontsize = 6)
      # Taille des labels
      plt.xticks(fontsize = 5)
      plt.yticks(fontsize = 5) 
      st.pyplot(plt)
    with col3:
      ax = sns.lmplot(x ='view', y ="addtocart", data = df_final_ag, order = 2, height = 5, aspect = 1)
      plt.title("Relation nombre de vues et nombre d'ajouts au panier", color = 'white', fontsize = 8)
      plt.xlabel('Vues', fontsize = 6)
      plt.ylabel('Ajouts au panier', fontsize = 6)
      # Taille des labels
      plt.xticks(fontsize = 5)
      plt.yticks(fontsize = 5) 
      st.pyplot(plt)
  
  ####################################################################################################
  # PAGE END CACHE FUNCTIONS
  ####################################################################################################

  st.markdown("<h4 class = 'bordered_blue'>III - Datavisualisation</h4>", unsafe_allow_html = True)
  st.write("")
  st.markdown("<h5 class = 'bordered'>Distribution du nombre de vue / ajout / transaction des produits</h5>", unsafe_allow_html = True)
  st.write("")

  fig = make_subplots(rows=1, cols=3,
                    subplot_titles = ['Distribution des views','Distribution des ajouts paniers','Distribution des transactions'])

  fig.add_trace(go.Histogram(x = df_final_ag['view'],
                            marker_color = 'blue',
                            name = 'Vues',
                            marker_line = dict(width = 1, color = 'white'),
                            xbins = dict(
                              start = 0,
                              end = 30,
                              size = 1)),
                            1, 1)
  fig.add_trace(go.Histogram(x = df_final_ag['addtocart'],
                            marker_color = 'orange',
                            name = 'Ajout panier',
                            marker_line = dict(width = 1, color = 'white'),
                            xbins = dict(
                              start = 0,
                              end = 10,
                              size = 1)),
                            1, 2)
  fig.add_trace(go.Histogram(x = df_final_ag['transaction'],
                            marker_color = 'green',
                            name='Transaction',
                            marker_line = dict(width = 1, color = 'white'),
                            xbins = dict(
                              start = 0,
                              end = 10,
                              size = 0.5)),
                            1, 3)
  # Ajout legend sur les axes x et y  
  fig.update_xaxes(title_text = "Nombre de vues", row = 1, col = 1)
  fig.update_xaxes(title_text = "Nombre d'ajout au panier", row = 1, col = 2)
  fig.update_xaxes(title_text = "Nombre de transactions", row = 1, col = 3)

  fig.update_yaxes(title_text = "Nombre d'observations", row = 1, col = 1)
  fig.update_yaxes(title_text = "Nombre d'observations", row = 1, col = 2)
  fig.update_yaxes(title_text = "Nombre d'observations", row = 1, col = 3)

  st.plotly_chart(fig, use_container_width = True)

  st.write("Au regard de ces distributions, nous pouvons dire que sur la période considérée, la plupart des produits ont été vu une fois, certains entre deux et cinq fois.")
  st.write("Le nombre d'ajout au panier est d'un produit voir deux donc assez faible par rapport au non ajout.")
  st.write("Il en est de même pour les transactions, 74k de non transaction vs 1.6k de transacttion")

  st.markdown("<h5 class = 'bordered'>Distribution du nombre de visiteurs pour un produit</h5>", unsafe_allow_html = True)
  st.write("")
  
  #Distribution du nbre de visiteurs pour un produit
  fig = go.Figure()
  fig.add_trace(go.Histogram(x = df_final_ag['visitorid'],
                            marker_color = 'blue',
                            name ='Nombre de visiteurs',
                            marker_line = dict(width = 1, color = 'white'),
                            xbins = dict(
                              start = 0,
                              end = 20,
                              size = 1))
                )
  fig.update_layout(showlegend = True)
  fig.update_layout(
                    xaxis_title="Nombre de visites",
                    yaxis_title="Nombre d'observations",
                    title="Duistribution du nombre de visiteurs"
                  )
  st.plotly_chart(fig, use_container_width = True)

  st.write("Il y a 39 217 visiteurs uniques qui se sont rendu une seule fois sur le site")
  st.write("Mais également un nombre non négligeable de visiteurs uniques qui se sont rendu deux fois ou plus sur le site")

  st.markdown("<h5 class = 'bordered'>Visualisation sur l'axe mensuel</h5>", unsafe_allow_html = True)
  st.write("")

  col1, col2, col3 = st.columns(3)
  with col1:
    fig1, ax1 = plt.subplots()
    ax1.pie(month_value_count_dff, explode = (0.1,0,0,0,0), labels = month_value_count_dff_labels, autopct = '%1.1f%%', textprops = {'color':"w", 'fontsize': 6})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Répartition des mois", color = "white", fontsize = 8)
    fig1.patch.set_facecolor('#0E1117')
    st.pyplot(fig1)
  with col2:
    fig1, ax1 = plt.subplots()
    ax1.pie(month_value_count_dffag, explode = (0.1,0,0,0,0), labels = month_value_count_dffag.keys(), autopct = '%1.1f%%', textprops = {'color':"w", 'fontsize': 6})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Nombre de mois unique", color = "white", fontsize = 8)
    fig1.patch.set_facecolor('#0E1117')
    st.pyplot(fig1)
  with col3:
    st.write("")

  st.write("Sur la période considérée, les mois de juin et juillet semble être la période qui a le plus d'évènements, contrairement au mois de septembre")
  st.write("Le nombre de mois unique où un produit est consulté ou acheté est de généralement de 1 voir 2 (donc un cycle d'achat relativement court ? / ou un produit qui n'est pas de 1ère nécessité ?)")
  st.write("Il est difficile d'apporter un avis métier au regard de l'anonymisation des produits")

  st.markdown("<h5 class = 'bordered'>Disponibilité du produit</h5>", unsafe_allow_html = True)
  st.write("")

  #Disponibilité du produit
  fig=go.Figure([go.Bar(x = availability_labels,
                        y = product_availability_values)])
  fig.update_layout(  title = 'Disponibilité des produits',
                      xaxis_title = 'Disponibilité',
                      yaxis_title = 'Nombre',
                      
                      )
  st.plotly_chart(fig, use_container_width = True)
  
  st.write("Sur la période considérée, nous avons un peu plus de produits indisponibles en comparaison aux produits disponibles, ce qui a dû influencer la non réalisation des transactions sur le site")
  
  st.markdown("<h5 class = 'bordered'>Catégories de produit</h5>", unsafe_allow_html = True)
  st.write("")

  col1, col2, col3 = st.columns(3)
  with col1:
    plt.figure(figsize = (5, 4), facecolor = '#0E1117')
    ax = plt.axes()
    plt.bar(product_values_topview.categoryid, product_values_topview['view'], color = "#636EFA")
    plt.xlabel('Catégories', color = "white", fontsize = 6)
    plt.ylabel('Nombre de vues', color = "white", fontsize = 6)
    plt.title('Top catégorie view', color = "white", fontsize = 6)
    # Taille des labels
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4) 
    # Couleur intérieur
    plt.rcParams['axes.facecolor'] = '#0E1117'
    # Couleur extérieur
    ax.set_facecolor("#0E1117")
    # Couleur barre des axes
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # Couleur label des axes
    ax.tick_params(axis ='x', colors = 'white')
    ax.tick_params(axis ='y', colors = 'white')
    st.pyplot(plt)
  with col2:
    plt.figure(figsize = (5, 3.9), facecolor = '#0E1117')
    ax = plt.axes()
    plt.bar(product_values_topadd.categoryid, product_values_topadd['addtocart'], color = "#EF553B")
    plt.xlabel('Catégories', color = "white", fontsize = 6)
    plt.ylabel("Nombre d'ajout au panier", color = "white", fontsize = 6)
    plt.title('Top catégorie ajouts paniers', color = "white", fontsize = 6)
    # Taille des labels
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4) 
    # Couleur intérieur
    plt.rcParams['axes.facecolor'] = '#0E1117'
    # Couleur extérieur
    ax.set_facecolor("#0E1117")
    # Couleur barre des axes
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # Couleur label des axes
    ax.tick_params(axis ='x', colors = 'white')
    ax.tick_params(axis ='y', colors = 'white')    
    st.pyplot(plt)
  with col3:
    plt.figure(figsize = (5, 3.9), facecolor = '#0E1117')
    ax = plt.axes()
    plt.bar(product_values_toptrans.categoryid, product_values_toptrans['transaction'], color = "#00CC96")
    plt.xlabel('Catégories', color = "white", fontsize = 6)
    plt.ylabel("Nombre de transactions", color = "white", fontsize = 6)
    plt.title('Top catégorie transactions', color = "white", fontsize = 6)
    # Taille des labels
    plt.xticks(fontsize = 4)
    plt.yticks(fontsize = 4) 
    # Couleur intérieur
    plt.rcParams['axes.facecolor'] = '#0E1117'
    # Couleur extérieur
    ax.set_facecolor("#0E1117")
    # Couleur barre des axes
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # Couleur label des axes
    ax.tick_params(axis ='x', colors = 'white')
    ax.tick_params(axis ='y', colors = 'white')    
    st.pyplot(plt)

  st.write("Le produit phare est le produit '1037' avec 26 transactions au total pour ce produit qui a été le plus acheté")
  
  st.markdown("<h5 class = 'bordered'>Relation entre : Vues et ajouts au panier / Ajouts et transactions / Visiteurs et vues</h5>", unsafe_allow_html = True)
  st.write("")

  st.image('./images/graph/related_view_add_trans01.png', width = 2350)
  st.markdown("<h5 class = 'bordered'>Relation linéaire entre : Visiteurs uniques et vues / Visiteur du produit et ajouts au panier / Vues et ajouts au panier</h5>", unsafe_allow_html = True)
  st.write("")

  st.image('./images/graph/related_view_add_visit01.png', width = 2300)

  st.markdown("<h5 class = 'bordered'>Correlation variables numerique</h5>", unsafe_allow_html = True)
  st.write("")
  
  col1, col2, col3 = st.columns(3)
  with col1:
    var_num = df_final_ag[['addtocart','transaction','view','visitorid', 'month']] 
    x=var_num.corr()
    fig, ax = plt.subplots(figsize = (8, 8), facecolor = '#0E1117')
    sns.heatmap(x, annot = True, ax = ax, cmap = 'plasma')
    # Taille des labels
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)     
    ax.tick_params(axis ='x', colors = 'white')
    ax.tick_params(axis ='y', colors = 'white')
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(colors = 'white',labelsize = 8)
    st.pyplot(plt)
  with col2:
    st.write("")
  with col3:
    st.write("")

  st.write("De façon logique, il y a une correlation entre le nombre de visiteur et le nombre de vues.")
  st.write("Celle entre le nombre de visiteurs et la transaction semble assez faible")
  
###########################################################################################################################################################################################################
# MODELE ML SUPERVISE ET NON SUPERVISE
###########################################################################################################################################################################################################
if page == pages[4] : 
  ####################################################################################################
  # PAGE BEGIN CACHE FUNCTIONS
  ####################################################################################################
  @st.cache_data
  def create_df_numeric(df_final_ag):
    # Sélectionner uniquement les colonnes numériques
    df_numeric = df_final_ag.drop(['available', 'categoryid','parentid'], axis = 1)
    return df_numeric

  df_numeric = create_df_numeric(df_final_ag)

  @st.cache_data
  def create_df_normalized(df_numeric):    
    df_normalized = df_numeric.copy()
    # Normaliser les données
    scaler = StandardScaler()
    col = ['addtocart', 'transaction', 'view', 'month', 'visitorid']
    df_normalized.loc[:,col] = scaler.fit_transform(df_normalized[col])    
    return df_normalized
  
  df_normalized = create_df_normalized(df_numeric)

  inertie = []
  distorsion = []
  silhouettes = []
  
  @st.cache_resource
  def ml_kmeans(df_normalized): #, inertie, distorsion, silhouettes):  

    K = range(2, 10)
    
    for k in K:
      kmeanModel = KMeans(n_clusters = k, random_state = 42, n_init = 10)
      kmeanModel.fit(df_normalized)
      inertie.append(kmeanModel.inertia_)
      distorsion.append(sum(np.min(cdist(df_normalized, kmeanModel.cluster_centers_, 'euclidean'), axis = 1)) / np.size(df, axis = 0))
      silhouettes.append(silhouette_score(df_normalized, kmeanModel.labels_))
    return inertie, distorsion, silhouettes
    
  inertie, distorsion, silhouettes = ml_kmeans(df_normalized)

  kmeans_output_results = pd.DataFrame(
    {'inertie': inertie,
     'distorsion': distorsion,
     'silhouettes': silhouettes
    })
  
  kmeans_output_results['index'] = np.arange(kmeans_output_results.shape[0])
  cols = kmeans_output_results.columns.tolist()
  cols = cols[-1:] + cols[:-1]
  kmeans_output_results = kmeans_output_results[cols]
  kmeans_output_results.set_index('index', inplace = True)

  @st.cache_resource
  def ml_kmeans_clf(df_normalized): #, inertie, distorsion, silhouettes):  
    #selon la méthode du coude, le nombre de cluster est de 4
    #Entrainement de l'algorithme sur le df, et calcul des positions des K centroïdes et les labels
    clf_kmean = KMeans(n_clusters = 4, random_state = 42, n_init = 10)
    clf_kmean = clf_kmean.fit(df_normalized)
    clf_kmean_label = clf_kmean.labels_
    clf_kmean_inertia = clf_kmean.inertia_
    clf_kmean_centroids = clf_kmean.cluster_centers_
    return clf_kmean, clf_kmean_label, clf_kmean_inertia, clf_kmean_centroids

  clf_kmean, clf_kmean_label, clf_kmean_inertia, clf_kmean_centroids = ml_kmeans_clf(df_normalized)

  @st.cache_resource
  def silhouette_sc(df_normalized, clf_kmean_label): #, inertie, distorsion, silhouettes):  

    silhouette_sc = silhouette_score(df_normalized,  clf_kmean_label)
    return silhouette_sc

  silhouette_sc = silhouette_sc(df_normalized, clf_kmean_label)

  @st.cache_data
  def df_final_ag_add_cluster(df_final_ag, clf_kmean_label, df_numeric):
    df_final_ag['cluster_label'] = clf_kmean_label
    # Interprétation des groupes
    # Afficher les statistiques des clusters
    cluster_stats = df_final_ag.groupby('cluster_label')[df_numeric.columns].mean()
    return df_final_ag, cluster_stats

  df_final_ag, cluster_stats = df_final_ag_add_cluster(df_final_ag, clf_kmean_label, df_numeric)

  # Créer une table de contingence entre cluster_label et disponibilité du produit
  @st.cache_data
  def contingency_table2_create(df_final_ag):
      contingency_table2 = pd.crosstab(df_final_ag['cluster_label'], df_final_ag['available'], normalize=0)*100
      return contingency_table2

  contingency_table2 = contingency_table2_create(df_final_ag)

  # Filtrage df avec au moins une transaction
  @st.cache_data
  def df_final_ag_trans_create(df_final_ag):
    df_final_ag_trans = df_final_ag.loc[df_final_ag['transaction']!=0]
    return df_final_ag_trans
  
  df_final_ag_trans = df_final_ag_trans_create(df_final_ag)

  # Création d'une table de contingence entre cluster_label et transaction
  @st.cache_data
  def contingency_table_create(df_final_ag_trans):
    contingency_table = pd.crosstab(df_final_ag_trans['cluster_label'], df_final_ag_trans['transaction'])
    return contingency_table
  
  contingency_table = contingency_table_create(df_final_ag_trans)

  # Récuprer les categories de produits qui sont dans le cluster 1
  @st.cache_data
  def categ_group_create(df_final_ag):
    dictag = {'categoryid' : 'unique'}
    categ_group = df_final_ag.groupby('cluster_label').agg(dictag)
    return categ_group
  
  categ_group = categ_group_create(df_final_ag)

  @st.cache_data
  def cluster1_category_count_create(categ_group):
    cluster1_category_count = len(categ_group.iloc[1,0])
    return cluster1_category_count
  
  cluster1_category_count = cluster1_category_count_create(categ_group)

  @st.cache_data
  def cluster3_category_count_create(categ_group):
    cluster3_category_count = len(categ_group.iloc[3,0])
    return cluster3_category_count
  
  cluster3_category_count = cluster3_category_count_create(categ_group)

  @st.cache_data
  def clf_kmean_centroids_0_create(clf_kmean_centroids):
    clf_kmean_centroids_0 = clf_kmean_centroids[:, 0]
    return clf_kmean_centroids_0

  clf_kmean_centroids_0 = clf_kmean_centroids_0_create(clf_kmean_centroids)

  @st.cache_data
  def clf_kmean_centroids_1_create(clf_kmean_centroids):
    clf_kmean_centroids_1 = clf_kmean_centroids[:, 1]
    return clf_kmean_centroids_1

  clf_kmean_centroids_1 = clf_kmean_centroids_1_create(clf_kmean_centroids)

  @st.cache_data
  def df_regression_create(df_numeric):    
    features_for_regression = ['addtocart', 'view', 'available', 'categoryid', 'parentid', 'month', 'visitorid']
    target_for_regression = ['transaction']
    df_regression = df_final_ag[features_for_regression + target_for_regression]
    return df_regression
  
  df_regression = df_regression_create(df_numeric)

  ####################################################################################################
  # PAGE END CACHE FUNCTIONS
  ####################################################################################################

  st.markdown("<h4 class = 'bordered_blue'>IV - Machine learning</h4>", unsafe_allow_html = True)
  st.write("")
  def ml_to_use(methode):
    if methode == 'Sélection modèle':
      st.markdown("<h5 class = 'bordered'>Constat</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("Après notre étape de Data Cleaning, Feature Engineering et de Preprocessing la Dataviz nous a procuré quelques indices comme la relation entre « view » et « addtocart » mais" +
               "l'anonymisation des données nous aveugle en grande partie et restreint nos trames d'études")
      st.write("")
      st.write("Donc nos tests de différents modèles de ML ont un seul axe principal :") 
      st.write("Comment augmenter la valeur « transaction » soit le nombre de vente online et pour cela nous avons appliqué de nombreux algorithmes pour répondre.")
      st.write("")
      st.write("Nous avons commencé par tester les modèles les plus simples et les plus rapides en termes de temps de calcul.") 
      st.write("De manière générale, il a fallu souvent faire un arbitrage entre le temps d'entraînement, la performance et l'interprétabilité des algorithmes.")

      st.markdown("<h5 class = 'bordered'>Modèles testés</h5>", unsafe_allow_html = True)
      st.write("")
      st.markdown("<h5 class = 'underlined'>Modèles Supervisé via la présence d'une variable cible et à partir de données labellisées</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("Nous cherchons à prédire une variable avec soit :")
      st.markdown("<span style='font-weight: bold;'>Régression :</span> prédire une variable de type quantitatif ( = valeurs continues )", unsafe_allow_html = True)
      st.markdown("<span style='font-weight: bold;'>Classification :</span> prédire une variable de type qualitatif ( = classe discrètes )", unsafe_allow_html = True)
      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Régression linéaire (Régression) :</span>", unsafe_allow_html = True)
      st.write("Sur cet ensemble de données, la régression linéaire peut prédire l'impact direct des variables comme 'view' et 'addtocart' sur les ventes ('transaction')." +
               "Son application ici a servi à évaluer l'influence linéaire et continue de ces caractéristiques sur la probabilité de conversion.")
      
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Forêt aléatoire (Régression) :</span>", unsafe_allow_html = True)
      st.write("La forêt aléatoire a été utilisée pour modéliser les relations complexes entre toutes les caractéristiques et les transactions, en tenant compte des interactions et de la non-linéarité." +
               "C'est un modèle robuste aux données aberrantes et peut améliorer l'exactitude des prédictions en réduisant le surajustement.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Régression logistique (Classification) :</span>", unsafe_allow_html = True)
      st.write("La régression logistique convient pour classifier les événements de 'transaction' en catégories binaires (achat ou pas). Elle peut révéler la probabilité d'achat en fonction des variables" +
               "explicatives et aider à cibler les interventions pour améliorer les conversions.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Arbre de décision (Classification) :</span>", unsafe_allow_html = True)
      st.write("L'arbre de décision facilitera la compréhension des chemins décisionnels menant à une vente, et identifiera les seuils critiques pour les variables comme 'addtocart'. Il peut servir à simplifier" +
               "la stratégie de targeting en isolant les facteurs décisifs.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>K-plus proches voisins (k-NN) (Classification) :</span>", unsafe_allow_html = True)
      st.write("Le modèle k-NN est choisi pour classifier les visiteurs en se basant sur la similarité de comportement. En identifiant les 'voisins' de transactions réussies, il pourrait aider à prédire et influencer les" +
               "comportements d'achat futurs en fonction de données comportementales similaires.")

      st.write("")
      st.markdown("<h5 class = 'underlined'>Modèles Non supervisé sans variable cible à prédire. On cherche à découvrir les structures sous-jacentes au sein de la base de données</h5>", unsafe_allow_html = True)
      st.write("")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>DBScan :</span>", unsafe_allow_html = True)
      st.write("DBScan appliqué à ces données, il peut détecter des groupes de comportements d'achat ou des anomalies sans à priori sur la forme des clusters, ce qui est utile pour comprendre des modèles d'achat non conventionnels et améliorer le ciblage marketing.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>KMeans :</span>", unsafe_allow_html = True)
      st.write("KMeans sera utilisé pour segmenter les visiteurs en groupes homogènes selon leur activité sur le site, ce qui peut informer des stratégies de marketing différenciées et améliorer l'allocation des ressources promotionnelles.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Clustering hiérarchique :</span>", unsafe_allow_html = True)
      st.write("Le clustering hiérarchique organisera les données en une structure arborescente, permettant de visualiser et de décider des niveaux d'agrégation pertinents pour les différentes stratégies de segmentation et de personnalisation des recommandations.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>Isolation Forest :</span>", unsafe_allow_html = True)
      st.write("Isolation Forest détectera des points de données atypiques dans le comportement des visiteurs, ce qui aidera à isoler les anomalies pouvant représenter des comportements d'achat frauduleux ou des erreurs, et affiner ainsi l'analyse.")

      st.write("")
      st.markdown("<span style='font-weight: bold; font-style: italic;'>One Class SVM :</span>", unsafe_allow_html = True)
      st.write("One Class SVM sera utile pour modéliser ce qui est considéré comme un comportement 'normal' et identifier les écarts, permettant de découvrir des opportunités ou des défis cachés dans les parcours d'achat des visiteurs.")

      st.write("")
      st.markdown("<h5 class = 'bordered'>Choix des modèles pertinents</h5>", unsafe_allow_html = True)
      st.write("")

      st.markdown("Suite à cet inventaire, nous avons pour objectif le choix d'un modèle pertinent en vue d'accroitre les performances de ce site internet mais au vu du problème de manque d'informations et en fonction des performances obtenues," +
                  "nous vous présentons 3 modèles ML : <span style='font-weight: bold;'>Regression Linéaire, KMeans et Isolation Forest</span>", unsafe_allow_html = True)
      st.write("")
      st.markdown("<h5 class = 'underlined'>Pourquoi ces 3 choix ?</h5>", unsafe_allow_html = True)
      st.markdown("- La <span style='font-weight: bold;'>Regression Linéaire</span> pour chercher à établir une relation linéaire entre une variable, dite expliquée ( Transactions ), et plusieurs variables, dites explicatives ( itemid, addtocart," +
                  "view, available, categoryid, parentid, month, visitorid ) pour établir des prédictions de vente.", unsafe_allow_html = True)
      st.markdown("- <span style='font-weight: bold;'>KMeans</span> avec son approche par clusters pour déterminer les catégories de produits cibles et les personnes susceptibles de réaliser des transactions sur ce site web.", unsafe_allow_html = True)
      st.markdown("- Enfin, comme notre jeu de données est déséquilibré, nous avons choisi <span style='font-weight: bold;'>Isolation Forest </span> conçu pour identifier les observations qui s'écartent de la norme.", unsafe_allow_html = True)
      st.write("")
      st.markdown("<span style='font-style: italic;'>Par ailleurs, nous avons testé plusieurs autres modèles ainsi que les techniques de rééchantillonnage sans succès. Vous les trouverez dans la section Autres modèles dans le menu 'Sélection modèle</span>", unsafe_allow_html = True)
      st.write("")
      st.write("")

    elif methode == 'Régression Linéaire':
      # Régression linéraire
      st.markdown("<h5 class = 'bordered_green'>Modèle de Régression Linéaire</h5>", unsafe_allow_html = True)
      st.write("")

      st.markdown("<h5 class = 'bordered'>Résultats Régression Linéaire</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("La régression linéaire appliquée aux données révèle un faible R² de 0.1249, indiquant que le modèle ne capture qu'une petite partie de la variance des transactions. Toutefois, les faibles valeurs de MSE, RMSE et MAE" +
               "(MSE de 0.0376 et RMSE de 0.1938) suggèrent que les erreurs de prédictions sont relativement basses. Cela pourrait indiquer que, bien que le modèle ne soit pas fortement prédictif des transactions en raison de la nature" +
               "simpliste des variables, il est cohérent dans ses prédictions. En pratique, la régression linéaire peut être utilisée pour estimer l'effet des vues et des ajouts au panier sur les ventes, mais d'autres facteurs non inclus" +
               "dans le modèle pourraient être nécessaires pour améliorer la prédiction. Nous l'évoquerons en conclusion.")
      st.write("")

      st.markdown("<h5 class = 'bordered'>Création d'un DataFrame pour la Régression linéraire</h5>", unsafe_allow_html = True)
      st.write("")

      st.write("Nous souhaitons prédire 'transaction' car il est crucial de choisir une variable cible qui varie de manière continue")

      st.write("Inclure 'transaction' en tant que colonne cible dans df_regression")
      st.write("Inclusion de la variable cible dans le DataFrame")
      st.write(df_regression)      
      st.markdown("""---""")
      st.markdown("<h5 class = 'bordered'>1 - Préparation des données</h5>", unsafe_allow_html = True)
      st.write("Séparer les caractéristiques (X) et la cible (y)")
      X = df_regression.drop('transaction', axis = 1)
      y = df_regression['transaction']

      col1, col2 = st.columns(2)
      with col1:
        st.write(X)
      with col2:
        st.write(y)
      st.markdown("""---""")
      st.write("Division des données en ensembles d'entraînement et de test")
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

      st.write("X_train, X_test")
      col1, col2 = st.columns(2)
      with col1:
        st.write(X_train)
      with col2:
        st.write(X_test)
      st.markdown("""---""")
      st.write("y_train, y_test")
      col1, col2 = st.columns(2)
      with col1:
        st.write(y_train)
      with col2:
        st.write(y_test)
      st.markdown("<h5 class = 'bordered'>2 - Normalisation des caractéristiques</h5>", unsafe_allow_html = True)      
      st.write("La normalisation est particulièrement utile pour la régression linéaire et peut également bénéficier à la forêt aléatoire dans certains cas.")

      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      col1, col2 = st.columns(2)
      with col1:
        st.write(X_train_scaled)
      with col2:
        st.write(X_test_scaled)
      st.markdown("<h5 class = 'bordered'>3 - Construction et Entraînement du Modèle</h5>", unsafe_allow_html = True)      
      st.markdown("<h5>Entraînement et Évaluation sur l'Ensemble de Test</h5>", unsafe_allow_html = True)

      linear_model = LinearRegression()
      linear_model.fit(X_train_scaled, y_train)

      st.markdown("<span class = 'margin_code'>linear_model = LinearRegression()</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>linear_model.fit(X_train_scaled, y_train)</span>", unsafe_allow_html = True)

      st.markdown("<h5 class = 'bordered'>4 - Évaluation des Modèles</h5>", unsafe_allow_html = True)      
      st.write("Nous utilisons des métriques différentes pour évaluer les performances du modèle")

      st.markdown("<h5>Prédiction sur l'ensemble de test</h5>", unsafe_allow_html = True)
      y_pred_linear = linear_model.predict(X_test_scaled)
      st.markdown("<span class = 'margin_code'>y_pred_linear = linear_model.predict(X_test_scaled)</span>", unsafe_allow_html = True)

      st.markdown("<h5>Évaluation pour la régression linéaire</h5>", unsafe_allow_html = True)

      st.write("MSE : Le MSE (Mean Squared Error) mesure la moyenne des carrés des erreurs, c'est-à-dire la différence entre les valeurs observées et celles prédites par le modèle.")
      st.write("MSE:", mean_squared_error(y_test, y_pred_linear))
      
      st.write("RMSE (Root Mean Squared Error) : Le RMSE est simplement la racine carrée du MSE, ce qui rend l'erreur plus interprétable car elle est dans les mêmes unités que la variable cible.")
      st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_linear)))
      
      st.write("MAE (Mean Absolute Error) : Le MAE mesure la moyenne des erreurs absolues. Contrairement au MSE, le MAE n'élève pas les erreurs au carré, ce qui donne une meilleure idée des erreurs en termes réels.")
      st.write("MAE:", mean_absolute_error(y_test, y_pred_linear))

      st.write("R² Score (Coefficient of Determination) : Le R² Score mesure à quel point les variations de la variable cible peuvent être expliquées par les caractéristiques du modèle. Un score de 1 indique que le modèle explique parfaitement " +
               "toute la variabilité, tandis qu'un score de 0 indiquerait que le modèle n'explique pas du tout la variabilité.")
      st.write("R2 Score:", r2_score(y_test, y_pred_linear))

      st.markdown("<h5 class = 'bordered'>5 - Interpretation</h5>", unsafe_allow_html = True)
      st.write("Les résultats montrent des mauvaises performances du modèle sur la tâche de prédiction du nombre de transaction (`transaction`)")

      st.markdown("<h5>Régression Linéaire</h5>", unsafe_allow_html = True)

      st.write("MSE et RMSE sont relativement bas, ce qui indique une bonne précision du modèle.")
      st.write("MAE est également faible, suggérant que les erreurs moyennes de prédiction sont minimes.")
      st.write("R2 Score de 0.12 : Le modèle de régression linéaire explique environ 12% de la variance de la variable dépendante. Il reste 88% de la variance non expliquée, ce qui indique une faible capacité de prédiction (ou la présence de facteurs "+
               "(?) non pris en compte dans le modèle.)")

      st.markdown("<h5>Visualition de la régression linéaire</h5>", unsafe_allow_html = True)
      
      col1, col2, col3 = st.columns(3)
      with col1:
        fig = plt.figure(figsize = (10,10))
        plt.scatter(y_pred_linear, y_test, c = 'green')
        plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
        plt.xlabel("Predictions")
        plt.ylabel("Valeurs réélles")
        plt.title('Régression Linéaire pour les transactions')
        st.pyplot(plt)
      with col2:
        st.write("")
      with col3:
        st.write("")

    elif methode == 'Modèle Kmeans':
    #KMEANS
      plt.rcParams['axes.facecolor'] = '#0E1117'

      st.markdown("<h5 class = 'bordered_grey'>Modele Kmeans</h5>", unsafe_allow_html = True)
      st.write("")

      st.markdown("<h5 class = 'bordered'>Résultats KMeans</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("L'application du modèle KMeans montre une segmentation claire du comportement des utilisateurs en quatre clusters, avec une majorité écrasante d'observations dans le cluster 0. Ce modèle peut être utilisé pour différencier les " +
               "stratégies de marketing et de promotion, en se concentrant sur les caractéristiques distinctes de chaque cluster. ")
      st.write("Par exemple, les clusters 1 et 3, bien que moins nombreuses, pourraient représenter un segment de niche à cibler différemment. Les scores de silhouette suggèrent un nombre optimal de clusters, ce qui" +
               "pourrait être affiné avec d'autres métriques pour obtenir une segmentation plus précise des clients. Nous y reviendrons dans la dernière partie.")
      st.write("")
      st.markdown("<h5 class = 'bordered'>Détérmination des clusters</h5>", unsafe_allow_html = True)
      st.write("")
      col1, col2, col3 = st.columns(3)
      with col1:
        st.markdown("<h5 class = 'centered'>Méthode du coude affichant le nombre de cluster optimal</h5>", unsafe_allow_html = True)
      with col2:
        st.markdown("<h5 class = 'centered'>Le score de silhouette montrant le k optimal</h5>", unsafe_allow_html = True)
      with col3:
        st.markdown("<h5 class = 'centered'>Inertie en fonction du nombre de cluster</h5>", unsafe_allow_html = True)
      
      col1, col2, col3 = st.columns(3)
      with col1:
        matplotlib.rc('axes',edgecolor = 'w')
        plt.figure(figsize = (5, 3), facecolor = '#0E1117')
        plt.axes().tick_params(axis = 'both', colors = 'white')
        plt.plot(range(2, 10), distorsion, 'bx-')
        plt.xlabel('Nombre de Clusters K', color = "white", fontsize = 6)
        plt.ylabel('Distorsion SSW/(SSW+SSB', color = "white", fontsize = 6)
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5)         
        st.pyplot(plt)
      with col2:
        matplotlib.rc('axes',edgecolor = 'w')
        plt.figure(figsize = (5, 2.9), facecolor = '#0E1117')
        plt.axes().tick_params(axis = 'both', colors = 'white')
        plt.plot(range(2, 10), silhouettes, 'bx-')
        plt.xlabel('k', color = "white", fontsize = 6)
        plt.ylabel('Score de silhouette', color = "white", fontsize = 6)
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5)         
        st.pyplot(plt)
      with col3:
        matplotlib.rc('axes',edgecolor = 'w')
        plt.figure(figsize = (5, 3), facecolor = '#0E1117')
        plt.axes().tick_params(axis = 'both', colors = 'white')
        plt.plot(range(2, 10), inertie)
        plt.xlabel('Nombre de clusters', color = "white", fontsize = 6)
        plt.ylabel('Inertie', color = "white", fontsize = 6)
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5)         
        st.pyplot(plt)
      
      st.write("Selon la méthode du coude, le nombre de cluster optimal est : 3")

      st.markdown("<h5 class = 'bordered'>Entrainement de l'algorithme sur le df_normalized, et calcul des positions des K centroïdes et les labels</h5>", unsafe_allow_html = True)
      st.write("")
      #st.write("Inertie du Kmeans optimal :", clf_kmean_inertia)
      st.write("Score de silhouette du Kmeans optimal :", silhouette_sc)
      
      st.markdown("<h5 class = 'bordered'>Visualisation des groupes</h5>", unsafe_allow_html = True)
      st.write("")

      col1, col2, col3 = st.columns(3)
      with col1:
        st.markdown("<p style='text-align: center;'>Nous avons près de 60.000 observations dans le cluster 0, nous en avons peu d'observation dans le cluster 2, et très peu dans le 1</p>", unsafe_allow_html = True)
      with col2:
        st.markdown("<p style='text-align: center;'>Visualisation des groupes</p>", unsafe_allow_html = True)
      with col3:
        st.markdown("<p style='text-align: center;'>Visualisation du nombre de transaction >0 par cluster</p>", unsafe_allow_html = True)

      col1, col2, col3 = st.columns(3)
      with col1:
        matplotlib.rc('axes',edgecolor = 'w')
        plt.figure(figsize = (5, 3.87), facecolor = '#0E1117')
        plt.axes().tick_params(axis = 'both', colors = 'white')
        plt.hist(df_final_ag['cluster_label'])
        plt.title("Répartition du nombre d'observations par cluster", color = 'white', fontsize = 8)
        plt.xlabel('Cluster', color = 'white', fontsize = 6)
        plt.ylabel("Nombre d'observations", color = 'white', fontsize = 6)
        plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5) 
        st.pyplot(plt)
      with col2:
        matplotlib.rc('axes',edgecolor = 'w')
        plt.figure(figsize = (5, 3.575), facecolor = '#0E1117')
        plt.axes().tick_params(axis = 'both', colors = 'white')
        sns.scatterplot(data = df_final_ag, x = 'transaction', y = 'visitorid', hue ='cluster_label')
        plt.scatter(clf_kmean_centroids_0, clf_kmean_centroids_1, marker = "o", color = "blue", s = 30, linewidths = 1, zorder = 10)
        plt.title("Clusters K-means", color = 'white', fontsize = 8)
        plt.xlabel('Transaction', color = 'white', fontsize = 6)
        plt.ylabel('Visitorid', color = 'white', fontsize = 6)
        plt.legend(title = 'Cluster', fontsize = 6, labelcolor = 'white')
        l = plt.legend()
        for text in l.get_texts():
          text.set_color("white")
          text.set_fontsize(6)
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5) 
        st.pyplot(plt)
      with col3:
        plt.figure(figsize = (5, 2), facecolor = '#0E1117')
        contingency_table.plot(kind = 'bar', stacked = True)
        plt.title('Répartition des transaction par cluster', color = 'white', fontsize = 8)
        plt.xlabel('Cluster', fontsize = 7)
        plt.ylabel('Nombre de transaction', fontsize = 7)
        plt.legend(title = 'Transaction', fontsize = 6, labelcolor = 'white')
        plt.legend(facecolor='k', labelcolor='w')
        l = plt.legend()
        for text in l.get_texts():
          text.set_color("white")
          text.set_fontsize(6)
        # Taille des labels
        plt.xticks(fontsize = 5)
        plt.yticks(fontsize = 5) 
        st.pyplot(plt)

      st.markdown("<h5 class = 'bordered'>Interprétation des groupes</h5>", unsafe_allow_html = True)
      st.write("")

      st.write("Afficher les statistiques des clusters")
      st.write(cluster_stats)

      st.write("Nous avons le groupe 1 'Les Fidèles' : ce groupe a une moyenne de transaction de plus de 1 avec très peu de vues et d'ajouts au panier. On peut émettre l'hypothèse qu'il s'agit des produits usuels achetés par les clients fidèles au site")
      st.write("Nous avons le groupe 3 'Les prospects' : ce groupe a une moyenne de transaction proche de 1 donc susceptible de faire des transactions avec un nombre elévé de visiteurs & vues; un groupe à haut potentiel")

      st.markdown("<h5 class = 'bordered'>Disponibilité des produits auprès de chaque cluster et les catégories des clusters 1 et 3</h5>", unsafe_allow_html = True)
      st.write("")

      st.write("Affichage du df_final avec les 4 clusters")
      st.write(df_final_ag.head())
      st.write("Vérification de la disponibilité des produits auprès de ces clusters")
      st.write(contingency_table2)
      st.write("Catégories de produits de chaque cluster")
      st.write(categ_group)
      st.write("Nous avons :", cluster1_category_count, "catégories dans le cluster 1")
      st.write("Nous avons :", cluster3_category_count, "catégories dans le cluster 3")

    elif methode == 'Isolation Forest':
      st.markdown("<h5 class = 'bordered_orange'>Modele Isolation Forest</h5>", unsafe_allow_html = True)
      st.write("")
      st.markdown("<h5 class = 'bordered'>Résultats Isolation Forest</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("Isolation Forest présente une précision globale de 0.8189 mais montre un déséquilibre entre la précision et le rappel pour la classe minoritaire (1).")
      st.write("Avec une haute valeur de rappel et une faible précision pour cette classe, le modèle est capable de détecter une grande partie des anomalies (ou transactions potentielles), mais au prix d'un nombre élevé de faux positifs. ")
      st.write("")
      st.write("Ces informations peuvent être utilisées pour identifier et enquêter sur des comportements d'achat inhabituels, potentiellement en vue d'optimiser la conversion et de réduire les transactions." +
                "Toutefois, pour équilibrer le modèle, il pourrait être nécessaire d'ajuster le seuil de classification, introduire un mécanisme de pondération ou d'intégrer des techniques de rééchantillonnage.")
      st.write("")
      st.markdown("<h5 class = 'bordered'>Constat</h5>", unsafe_allow_html = True)
      st.write("")
      st.write("Notre jeux de données est très déséquilibré avec un classe majoritaire (0) très dominante et une deuxième classe très minoritaire donc nous allons utiliser l'Isolation Forest.")
      st.write("Ce modèle est conçu pour identifier les observations qui s'écartent de la norme, ce qui peut correspondre à notre classe minoritaire dans un contexte déséquilibré.")
      st.write("Nous allons d'abord entraîner les modèles sur une partie des données (ensemble d'entraînement) puis évaluer leur performance sur une autre partie (ensemble de test), et enfin évaluer leur performance sur l'intégralité des données pour comparer.")

      st.markdown("<h5 class = 'bordered'>1 - Préparation des données</h5>", unsafe_allow_html = True)
      st.markdown("<h5>Préparation des données</h5>", unsafe_allow_html = True)
      st.write("- Création de df_final_ag_clf pour classification")
      df_final_ag_clf = df_final_ag
      df_final_ag_clf.loc[df_final_ag_clf.transaction != 0] = 1

      st.markdown("<span class = 'margin_code'>df_final_ag_clf = df_final_ag</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>df_final_ag_clf.loc[df_final_ag_clf.transaction !=  0] = 1</span>", unsafe_allow_html = True)

      X = df_final_ag_clf.drop('transaction', axis=1)
      y = df_final_ag_clf['transaction']
      
      st.markdown("<span class = 'margin_code'>X = df_final_ag_clf.drop('transaction', axis = 1)</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>y = df_final_ag_clf['transaction']</span>", unsafe_allow_html = True)
      
      st.markdown("<h5>Séparation en ensembles d'entraînement et de test</h5>", unsafe_allow_html = True)

      X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size = 0.2, random_state = 42)
      st.markdown("<span class = 'margin_code'>X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size = 0.2, random_state = 42)</span>", unsafe_allow_html = True)

      st.markdown("<h5 class = 'bordered'>2 - Normalisation des caractéristiques</h5>", unsafe_allow_html = True)
      scaler = StandardScaler()
      X_train_rf_scaled = scaler.fit_transform(X_train_rf)
      X_test_rf_scaled = scaler.transform(X_test_rf)
      st.write("")
      st.markdown("<span class = 'margin_code'>scaler = StandardScaler()</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>X_train_scaled = scaler.fit_transform(X_train)</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>X_test_scaled = scaler.transform(X_test)</span>", unsafe_allow_html = True)

      col1, col2 = st.columns(2)
      with col1:
        st.write(X_train_rf_scaled)
      with col2:
        st.write(X_test_rf_scaled)

      st.markdown("<h5 class = 'bordered'>3 - Construction et Entraînement du Modèle</h5>", unsafe_allow_html = True)
      st.markdown("<h5>Entraînement et Évaluation sur l'Ensemble de Test</h5>", unsafe_allow_html = True)

      iso_forest = IsolationForest(n_estimators = 100, contamination ='auto', random_state = 42)
      iso_forest.fit(X_train_rf_scaled)

      st.markdown("<span class = 'margin_code'>iso_forest = IsolationForest(n_estimators = 100, contamination = 'auto', random_state = 42)</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>iso_forest.fit(X_train_rf_scaled)</span>", unsafe_allow_html = True)

      st.markdown("<h5 class = 'bordered'>4 - Évaluation des Modèles</h5>", unsafe_allow_html = True)      
      st.write("Nous utilisons des métriques différentes pour évaluer les performances du modèle")

      st.markdown("<h5>Prédiction sur l'ensemble de test</h5>", unsafe_allow_html = True)
      y_pred_rf_test_iso = iso_forest.predict(X_test_rf_scaled)

      st.markdown("<h5>Évaluation sur l'ensemble de test</h5>", unsafe_allow_html = True)
      st.write("Isolation Forest Performance sur le Test Set:")
      st.write("Accuracy:", accuracy_score(y_test_rf, y_pred_rf_test_iso))
      st.write(classification_report(y_test_rf, y_pred_rf_test_iso))

      st.markdown("<h5>Évaluation sur l'Intégralité des Données</h5>", unsafe_allow_html = True)
      X_rf_scaled_full = scaler.transform(X)
      y_pred_rf_full_iso = iso_forest.predict(X_rf_scaled_full)

      st.markdown("<span class = 'margin_code'>X_rf_scaled_full = scaler.transform(X)</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>y_pred_rf_full_iso = iso_forest.predict(X_rf_scaled_full)</span>", unsafe_allow_html = True)

      st.markdown("<h5>Convertir les prédictions pour correspondre aux étiquettes cibles (1 pour anomalies, 0 pour normal)</h5>", unsafe_allow_html = True)
      y_pred_rf_full_iso[y_pred_rf_full_iso == 1] = 0
      y_pred_rf_full_iso[y_pred_rf_full_iso == -1] = 1

      st.markdown("<span class = 'margin_code'>y_pred_rf_full_iso[y_pred_rf_full_iso == 1] = 0</span>", unsafe_allow_html = True)
      st.markdown("<span class = 'margin_code'>y_pred_rf_full_iso[y_pred_rf_full_iso == -1] = 1</span>", unsafe_allow_html = True)

      st.write("Isolation Forest Performance sur l'Intégralité des Données:")
      st.write("Accuracy:", accuracy_score(y, y_pred_rf_full_iso))
      st.write(classification_report(y, y_pred_rf_full_iso))

      st.markdown("<h5 class = 'bordered'>5 - Interpretation</h5>", unsafe_allow_html = True)
      st.markdown("<h5>Matrice de Confusion</h5>", unsafe_allow_html = True)
      st.write("La matrice de confusion pour chaque modèle sur l'ensemble de test peut nous fournir des insights visuels sur leur performance, en particulier la répartition des vrais positifs, faux positifs, vrais négatifs, et faux négatifs.")
      
      cm_iso = confusion_matrix(y_test_rf, y_pred_rf_test_iso)

      st.markdown("<span class = 'margin_code'>cm_iso = confusion_matrix(y_test_rf, y_pred_rf_test_iso)</span>", unsafe_allow_html = True)

      col1, col2, col3 = st.columns(3)
      with col1:
        fig = plt.figure(figsize = (10,10))
        sns.heatmap(cm_iso, annot = True, fmt = "d", cmap = "Blues", xticklabels = [0, 1], yticklabels=[0, 1])
        plt.title('Matrice de Confusion pour Isolation Forest')
        plt.xlabel('Prédictions')
        plt.ylabel('Valeurs réélles')
        st.pyplot(plt)
      with col2:
        st.write("")
      with col3:
        st.write("")
  
    elif methode == 'Résumé des modèle testés':
      st.markdown("<h5 class = 'bordered_purple'>Résumé des modèle testés</h5>", unsafe_allow_html = True)
      st.write("")

      st.image('./images/descr/resume_modeles_utilises.png', width = 2700)

      

  choix = ['Sélection modèle', 'Régression Linéaire','Modèle Kmeans', 'Isolation Forest', 'Résumé des modèle testés']
  option = st.selectbox('Détail par modèle', choix)

  ml_to_use(option)
  
###########################################################################################################################################################################################################
# Interprétation et conclusion
###########################################################################################################################################################################################################
if page == pages[5] : 
  st.markdown("<h4 class = 'bordered_blue'>V- Interprétation et conclusion</h4>", unsafe_allow_html = True)
  st.write("")
  st.markdown("<h5 class = 'bordered'>Synthèse des résultats de ces 3 modèles et nos recommandations pour ce site et son gestionnaire</h5>", unsafe_allow_html = True)
  st.write("")
  st.write("Il est rappelé que durant l'étape de Data visualisation sur la période considérée, il y a un peu plus de produits indisponibles en comparaison aux produits disponibles. Cela a une influence non-quantifié mais réelle sur la non réalisation de " +
           "transactions donc sur les résultats des modèles.")
  st.write("")
  st.markdown("Grâce à <span style='font-weight: bold;'>KMeans</span>, quatre groupes distincts de comportement utilisateurs ont été identifiés. Une attention devrait être portée sur les clusters 1 et 3 qui pourraient représenter des niches de marché des " +
              "opportunités de ventes additionnelles. Nous recommandons d'analyser les catégories de produits de ces clusters pour identifier des produits ou des offres qui ont une performance au-dessus de la moyenne.", unsafe_allow_html = True)
  st.write("")
  st.write("Le score de silhouette montre une segmentation acceptable.")
  st.write("Le propriétaire du site pourrait envisager des stratégies marketing personnalisées pour chaque cluster, comme des recommandations de produits ou des campagnes de réengagement basées sur les préférences de chaque segment. Il doit également " +
           "s'assurer de la disponibilité des produits qui pénalise le taux de conversion de prospects.")
  st.write("")
  st.markdown("<span style='font-weight: bold;'>Priorité 1 :</span> Nous avons le cluster 1 qui peut donc être isolé afin de mener des actions pour optimiser le parcours des clients sur le site, permettant théoriquement d'augmenter le taux de " +
              "conversion. Il faudra améliorer de façon significative le taux de conversion auprès de cette cible en rendant les produits disponibles sur le site (40% de produits étant indisponible sur ce cluster).", unsafe_allow_html = True)
  st.markdown("<span style='font-weight: bold;'>Priorité 2 :</span> Nous avons aussi le cluster 3 qui regroupe des acheteurs déjà fidèles au site auprès desquels on peut faire des actions promotionnelles pour augmenter les volumes d'achat.", unsafe_allow_html = True)
  st.write("")
  st.write("Les cluster 0 et 2 seraient les plus difficiles à atteindre avec un taux d'indisponibilité des produits qui monte à 60% auprès du cluster 0 par exemple.")
  st.write("Au-delà des actions marketing, la disponibilité des produits pose un vrai problème.")

  st.markdown("La <span style='font-weight: bold;'>RLin</span>a donné des performances intéressantes en termes de précision mais elle a été insuffisante sur la tâche de prédiction du nombre de transactions.", unsafe_allow_html = True)
  st.write("")
  st.write("Il est important de noter que ce constat (à cause du R2) doit être interprété en fonction du contexte et de l'objectif du modèle. La faible capacité de prédiction affiché peut être acceptable si le modèle est utilisé pour des prédictions " +
           "approximatives au vu des variables prises en compte et de leur qualité.")
  st.write("Ainsi en l'absence d'amélioration de la qualité de la source data, nous ne pouvons l'utiliser par défaut que pour des prévisions approximatives.")
  st.write("Pour le propriétaire du site, cela signifie que les facteurs actuellement suivis n'influencent les transactions que de manière limitée. Pour améliorer le modèle, il faudrait considérer d'autres variables, peut-être des données démographiques " +
           "ou des historiques de comportement des utilisateurs, pour affiner l'analyse. L'utilisation de la régression linéaire pourrait être maintenue pour évaluer l'impact de promotions ciblées sur le nombre de transactions.")
  st.write("La régression linéaire pourrait être affinée avec des variables supplémentaires pour mieux comprendre les facteurs affectant les ventes.")
  st.write("")
  st.markdown("Pour finir, <span style='font-weight: bold;'>L'IFor</span> semble performant pour identifier les observations normales mais beaucoup moins performant pour prédire précisément des anomalies (les ventes dans notre cas) sans générer un nombre " +
              "élevé de faux positifs.", unsafe_allow_html = True)
  st.write("")
  st.write("Il peut être affiné pour mieux distinguer les comportements normaux des anomalies, et potentiellement identifier des opportunités de ventes croisées ou de promotions ciblées sur les utilisateurs dont le comportement s'écarte de la norme.")
  st.write("")
  st.markdown("<h5 class = 'bordered'>Nos préconisations</h5>", unsafe_allow_html = True)
  st.write("")
  st.write("La stratégie globale devrait inclure :")
  st.markdown("<span style='margin-left: 30px;'>•	L'utilisation de la régression linéaire pour évaluer l'effet des actions marketing sur les ventes.</span>", unsafe_allow_html = True)
  st.markdown("<span style='margin-left: 30px;'>•	L'application des insights du KMeans pour personnaliser l'expérience utilisateur et les offres promotionnelles par cluster de comportement d'achat, si c'est un site avec une market place, sensibilisez " +
              "les vendeurs sur l'impact de l'indisponibilité des produits sur le site (avec les données chiffrées du Kmeans).</span>", unsafe_allow_html = True)
  st.markdown("<span style='margin-left: 30px;'>•	L'intégration d'Isolation Forest pour surveiller et agir sur les ventes.</span>", unsafe_allow_html = True)
  st.write("")
  st.markdown("<span style='margin-left: 30px;'>•	La synergie de ces approches permettra une compréhension approfondie des comportements utilisateurs et une amélioration de l'efficacité des campagnes marketing.</span>", unsafe_allow_html = True)
  st.markdown("<span style='margin-left: 30px;'>•	Une veille sur l'introduction de nouvelles variables et un raffinement continu des modèles sont conseillés pour rester compétitif sur le marché.</span>", unsafe_allow_html = True)
  
  st.markdown("<span style='font-style: italic;'>Pour aller plus loin si nous avions accès à des données non anonymisées, l'objectif serait d'inclure des données socio-démographiques…ou réaliser un entretien avec l'équipe commande du site Web ou celle du " +
              "stock au vu de sa situation.</span>", unsafe_allow_html = True)
