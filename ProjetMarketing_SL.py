import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix
from scipy.stats import chi2_contingency


@st.cache_data
def load_data():
    return pd.read_csv('bank.csv')


def partie_0(df=None):
    st.write("PARTIE 0 : Chargement du fichier et Introduction du projet/Sujet")
    # Ici, vous pouvez introduire votre projet, montrer un aperçu des données, etc.
    if df is None:
        df = load_data()
    
    st.write(df.head())
    return df

def partie_1(df):
    st.write("PARTIE 1 : ANALYSE GLOBALE DU JEU DE DONNEES")
    st.write("Visualisation des premières lignes :")
    st.write(df.head())

    st.write("Informations sur le dataframe :")
    st.write(df.info())

    st.write(f"Le dataframe contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")

    st.write("Statistiques des colonnes numériques :")
    st.table(df.describe())

    st.write(f"Il y a {df.duplicated().sum()} doublons dans le dataframe.")

    st.write("Valeurs manquantes par colonne :")
    st.write(df.isna().sum())

    st.write("Nombre de modalités par champ :")
    st.write(df.nunique())

    cat_vars = df.select_dtypes(include=['object','category']).columns
    num_vars = df.select_dtypes(include=['float64', 'int64']).columns
    st.write(f"Les variables catégorielles sont : {cat_vars}")
    st.write(f"Les variables continues sont : {num_vars}")

def partie_2(df):
    st.write("PARTIE 2 : TRAITEMENT DES DONNEES")
    # Votre code pour cette section

def partie_3(df):
    st.write("PARTIE 3 : ANALYSE DES VARIABLES")
    st.subheader("Analyse des Variables")

    cat_vars = df.select_dtypes(include=['object', 'category']).columns
    num_vars = df.select_dtypes(include=['float64', 'int64']).columns

    # Pearson pour variables continues
    if st.checkbox('Pearson pour variables continues'):
        st.write("\n Partie /////Pearson///// pour variables continues************\n")
        corr_cat = df[num_vars].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr_cat, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap - Corrélation entre variables Continues")
        st.pyplot(fig)

    # ANOVA pour variables catégorielles vs continues
    if st.checkbox('ANOVA pour variables catégorielles vs. continues'):
        st.write("\n Partie /////ANOVA/////  pour variables catégorielles**********************\n")
        results_anova = []
        for cat_var in cat_vars:
            for cont_var in num_vars:
                formula = f"{cont_var} ~ C({cat_var})"
                model = ols(formula, data=df).fit()
                table = sm.stats.anova_lm(model, typ=2)
                p_val = table.loc["C("+cat_var+")", "PR(>F)"]
                if p_val < 0.05:
                    results_anova.append([cat_var, cont_var, "+++Correlées+++"])
                else:
                    results_anova.append([cat_var, cont_var, "NON correlés---"])
        results_anova = pd.DataFrame(results_anova, columns=["Variable catégorielle", "Variable continue", "Résultat test"])
        st.write(results_anova)

    # Khi2 et V Cramer pour variables catégorielles
    if st.checkbox('Khi2 et V Cramer pour variables catégorielles'):
        st.write("\n  Partie ////// Khi2 et Vcrammer  ///// pour variables catégorielles**************************\n")
        Nb = df.shape[0]
        results_khi2 = []
        for cat_var in cat_vars:
            for cat_var2 in cat_vars:
                p_chi2, coef = chi2_test(df[cat_var], df[cat_var2], Nb)
                if p_chi2 < 0.05:
                    results_khi2.append([cat_var, cat_var2,"+++Correlées+++ {:.4f} V_Kramer={:.2f}".format(p_chi2,coef)])
                else:
                    results_khi2.append([cat_var, cat_var2,"NON correlés--- {:.4f}".format(p_chi2)])
        results_khi2 = pd.DataFrame(results_khi2, columns=["Variable catégorielle 1", "Variable catégorielle 2","Test Khi2"])
        st.write(results_khi2)

def chi2_test(col1, col2, Nb):
    contingency_table = pd.crosstab(col1, col2)
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    cramer_v = V_Cramer(contingency_table, Nb)
    return p, cramer_v

def V_Cramer(tab, N):
    qui2 = pd.DataFrame(chi2_contingency(tab), index=["chi2","p_value","degree_of_freedom","expected"], columns=["resultat"]).loc['chi2'][0]
    k, r = tab.shape
    r_ = r - (((r - 1) ** 2) / (N - 1))
    k_ = k - (((k - 1) ** 2) / (N - 1))
    phi2 = max(0, (qui2 / N) - ((k - 1) * (r - 1) / (N - 1)))
    V = np.sqrt(phi2 / min(k_ - 1, r_ - 1))
    return V





def partie_4(df):
    st.write("PARTIE 4 : DATA VIZ - EXPLORATION DES DONNEES")
    st.subheader("Exploration des Données")

    var1 = num_vars.drop(['balance','duration', 'pdays'])
    var2 = ['balance']
    var3 = ['duration', 'pdays']

    fig, axs = plt.subplots(ncols=3, figsize=(20, 20))
    df[var1].boxplot(ax=axs[0], rot=45, showfliers=False)
    df[var2].boxplot(ax=axs[1], rot=45, showfliers=False)
    df[var3].boxplot(ax=axs[2], rot=45, showfliers=False)
    plt.title("Boîtes à moustache sur les variables numériques")
    st.pyplot(fig)

    # Les autres visualisations
    st.write("Relation entre chaque paire de variables")
    st.write(sns.pairplot(df, diag_kind='kde'))

    st.write("Relation entre paire de variable quanti")
    st.write(sns.pairplot(df[num_vars], diag_kind='kde'))

    df_grouped = df[['deposit','pdays']]
    df_grouped['pdays_grouped'] = pd.cut(df['pdays'], bins=range(0, 366, 3))
    df_grouped = df_grouped.groupby('pdays_grouped')['deposit'].value_counts().unstack().reset_index()
    df_grouped.rename(columns={True: 'True', False: 'False'}, inplace=True)
    df_grouped['Percentage_True'] = df_grouped['True'] / (df_grouped['True'] + df_grouped['False']) * 100

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.barplot(data=df_grouped, x='pdays_grouped', y='True', label='True', color='blue')
    sns.barplot(data=df_grouped, x='pdays_grouped', y='False', label='False', color='red', bottom=df_grouped['True'])
    for index, row in df_grouped.iterrows():
        plt.text(index, row['True'] + row['False'] + 1, f"{row['Percentage_True']:.0f}%", ha='center', fontsize=10)

    plt.title("Répartition 'pdays' % fonction de 'deposit'")
    plt.xlabel("pdays")
    plt.ylabel("Pourcentage")
    plt.xticks(rotation=45)
    plt.legend(title='deposit')
    st.pyplot(fig)

    # Display pdays efficaces
    st.write(df_grouped.sort_values(by='Percentage_True', ascending=False).head(5))

    # Diagrammes circulaires
    # Pour 'job'
    job_counts = df['job'].value_counts()
    fig, ax = plt.subplots()
    plt.pie(job_counts, labels=job_counts.index, autopct='%1.1f%%', startangle=90, shadow=True, explode=[0.05]*len(job_counts))
    plt.title('Répartition des clients selon leur profession')
    plt.axis('equal')
    st.pyplot(fig)

    # Pour 'type_job'
    type_job_counts = df['type_job'].value_counts()
    fig, ax = plt.subplots()
    plt.pie(type_job_counts, labels=type_job_counts.index, autopct='%1.1f%%', startangle=90, shadow=True, explode=[0.05]*len(type_job_counts))
    plt.title('Répartition des clients selon leur TYPE de profession')
    plt.axis('equal')
    st.pyplot(fig)

    # Pour 'education'
    edu_counts = df['education'].value_counts()
    fig, ax = plt.subplots()
    plt.pie(edu_counts, labels=edu_counts.index, shadow=True, autopct='%1.1f%%', startangle=90, explode=[0,0,0.1])
    plt.title('Répartition des clients selon leur niveau d\'éducation')
    plt.axis('equal')
    st.pyplot(fig)

    # Répartition dépôt par mois
    df_bank = df.sort_values('month_number')
    df_bank['month_chart'] = df_bank['month_number'].apply(lambda x: calendar.month_name[int(x)])
    ordered_months = [calendar.month_name[i] for i in range(1, 13)]
    df_bank['month_chart'] = pd.Categorical(df_bank['month_chart'], categories=ordered_months, ordered=True)
    df_bank = df_bank.sort_values('month_chart')

    fig, axs = plt.subplots(figsize=(12, 6))
    sns.histplot(df_bank, x='month_chart', hue='deposit', binwidth=120, kde=True, ax=axs, multiple='stack')
    axs.set_xlabel('Mois')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')
    axs.set_title('Répartition dépôt par mois')
    st.pyplot(fig)

    # RATIO Répartition dépôt par mois
    df_monthly = df_bank.groupby(['month_chart', 'deposit'])['deposit'].count().unstack('deposit')
    df_monthly['deposit_ratio'] = df_monthly[True] / (df_monthly[True] + df_monthly[False])
    df_monthly = df_monthly.loc[ordered_months].reset_index()

    fig, axs = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_monthly, x='month_chart', y='deposit_ratio', ax=axs, color='orange')
    axs.set_xlabel('Mois')
    axs.set_ylabel('Ratio de dépôts')
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')
    axs.set_title('RATIO Répartition dépôt par mois')
    st.pyplot(fig)

    st.write("""
    Les volumes sont plus importants en : Février, avril, mai et novembre.
    Mais à l'inverse, les taux de souscriptions sont meilleurs en : mars, entre juin et octobre ainsi qu'en décembre.
    L'efficacité de la campagne mériterait donc d'être améliorée pour augmenter le taux de succès là où les volumes sont présents.
    """)

def partie_5(df):
    st.write("PARTIE 5 : DATA VIZ - ZOOM SUR LES VARIABLES SELECTIONNEES POUR MODELISATION")
    # Votre code pour cette section

def partie_6(df):
    st.write("PARTIE 6 : MODELISATION")
    # Votre code pour cette section

def partie_7(df):
    st.write("PARTIE 7 : OPTIMISATIONS DES MODELES RETENUS")
    # Votre code pour cette section

def partie_8(df):
    st.write("PARTIE 8 : REENTRAINEMENT DU JEU AVEC VARIABLES IMPORTANTES")
    # Votre code pour cette section

def partie_9(df):
    st.write("PARTIE 9 : CONCLUSION")
    st.write("On doit prédire le succès de la campagne sur notre base client.")
    # Votre code pour cette section

def main():
    st.title("Analyse des données bancaires")
    
    menu = [
        "PARTIE 0 : Chargement du fichier et Introduction du projet/Sujet",
        "PARTIE 1 : ANALYSE GLOBALE DU JEU DE DONNEES",
        "PARTIE 2 : TRAITEMENT DES DONNEES",
        "PARTIE 3 : ANALYSE DES VARIABLES",
        "PARTIE 4 : DATA VIZ - EXPLORATION DES DONNEES",
        "PARTIE 5 : DATA VIZ - ZOOM SUR LES VARIABLES SELECTIONNEES POUR MODELISATION",
        "PARTIE 6 : MODELISATION",

        "PARTIE 7 : OPTIMISATIONS DES MODELES RETENUS",
        "PARTIE 8 : REENTRAINEMENT DU JEU AVEC VARIABLES IMPORTANTES",
        "PARTIE 9 : CONCLUSION"
    ]
    
    choice = st.sidebar.radio("Menu principal", menu)

    if choice == menu[0]:
        uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")
        if uploaded_file:
            df = load_data(uploaded_file)
            partie_0(df)
        else:
            partie_0()

    elif choice == menu[1]:
        try:
            partie_1(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")
    
    elif choice == menu[2]:
        try:
            partie_2(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")

    elif choice == menu[3]:
        try:
            partie_3(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")

    elif choice == menu[4]:
        try:
            partie_4(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")


    elif choice == menu[5]:
        try:
            partie_5(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")


    elif choice == menu[6]:
        try:
            partie_6(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")

    elif choice == menu[7]:
        try:
            partie_7(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")


    elif choice == menu[8]:
        try:
            partie_8(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")
    
    elif choice == menu[8]:
        try:
            partie_8(df)
        except NameError:
            st.write("Veuillez d'abord charger les données dans la PARTIE 0.")


if __name__ == "__main__":
    main()
