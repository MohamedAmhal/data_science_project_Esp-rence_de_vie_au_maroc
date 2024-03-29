# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:37:33 2023

@author: MOHAMED AMHAL
"""
#les biblio:
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re





#la biblio de sepearations des bases de données de trining et testes

from sklearn.model_selection import train_test_split





#modele 1: regression linéaire:(les biblios)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





#modele 2 : SVM (les biblios)

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler




#modele 3 : Random forest: (les biblios)

from sklearn.ensemble import RandomForestRegressor





#modele 4 :Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor





###############################################################################
#                                                                             #
#                                                                             #
#                   Le prétraitement des données(DATA CLEANING)               #                                                         #
#                                                                             #
#                                                                             #
###############################################################################




#importer la base de données :
    
df = pd.read_csv(r"C:\Users\Surface\OneDrive\Bureau\life_expectancy.csv",encoding='ISO-8859-1')



#afficher les informations de chaque colonne:
    
df.info()



#voir les statistiques des variables quantitatifs:
    
stat_quantita = df.describe()



#voir les statistiques des variables qualitatifs:
    
stat_qualita = df.describe(include='object')



#voir le nombre des valeurs manquantes de chaque varaible :
    
valeur_manqq = df.isnull().sum().sort_values(ascending = False)




#diviser la base de donnes en deux types : qualitatif et quantitatif:
    
base_qualitatif = []
base_quantitatif = []




#df.dtype => colonne,type de donnees enumerate =>pour avoir un dictionnaire

for i,j in enumerate(df.dtypes):
    if j == 'object':
        base_qualitatif.append(df.iloc[:,i]) #selectionner tt la colonne
    else:
        base_quantitatif.append(df.iloc[:,i])
        
        

#transformer les listes en dataframe (transpose pour l'inversions des lignes des colonnes)

base_qualitatif = pd.DataFrame(base_qualitatif).transpose()
base_quantitatif = pd.DataFrame(base_quantitatif).transpose()




#remplacer les valeurs manquantes des varaiables qualitatif (mode) :
    
base_qualitatif = base_qualitatif.apply(lambda x : x.fillna(x.value_counts().index[0]))




#concatenations des mots des pays :
    
base_qualitatif['pays'] = base_qualitatif['pays'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)).replace(' ', '_') if x else '')




#transformer la variable categorique status en variable numerique :
#Developing = 0 et  Developed = 1

base_qualitatif['Statut'] = df['Statut'].apply(lambda x: 0 if x == 'Developing' else 1) 



#verification :
    
print(base_qualitatif.isnull().sum())   



#print(base_qualitatif['Statut'].value_counts())




#remplecer les valeurs manquantes des variables qualitatif par la moyenne arithmetique:
    
base_quantitatif = base_quantitatif.apply(lambda x : x.fillna(x.mean()))



#verification :
    
print(base_quantitatif.isnull().sum())




#base de donnes nett:
    
base = pd.concat([base_qualitatif,base_quantitatif],axis = 1).drop(["pays"],axis = 1)
base2 = pd.concat([base_qualitatif,base_quantitatif],axis = 1)




#concatenations des bases de donnees:
    
X = pd.concat([base_qualitatif,base_quantitatif],axis = 1).drop(['Espérance_de_vie'],axis = 1)
Y = base_quantitatif['Espérance_de_vie']





###############################################################################
#                                                                             #
#                                                                             #
#                ANALYSE EXPLORATOIRE(DATA VISUALISATION)                     #                                      #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################







    
# Histogramme de la variable Y "Espérance_de_vie":
    
plt.figure(figsize=(10, 6))
sns.histplot(base['Espérance_de_vie'], bins=30, kde=True)
plt.title("Distribution de l'espérance de vie")
plt.xlabel('Espérance_de_vie')
plt.ylabel('Fréquence')
plt.show()





#Matrice de correlation entre les variables numeriques :(voir les relation entre les variables )
    
correlation_matrix = base.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de correlation')
plt.show()





#Analyse de relations spécifiques (les tests croisées) :
    
    

     # La relation entre Espérance_de_vie et statut:
plt.figure(figsize=(10, 6))
sns.boxplot(x='Statut', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et statut')
plt.xlabel('Statut')
plt.ylabel('Espérance_de_vie')
plt.show()




     
#la relation entre Espérance_de_vie et alcool:
         
         
             #graghique linéaire :
                 
plt.figure(figsize=(10, 6))
sns.lineplot(x='Alcool', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et alcool')
plt.xlabel('Alcool')
plt.ylabel('Espérance_de_vie')
plt.show()




             #nuage de points:
                 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Alcool', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et alcool')
plt.xlabel('Alcool')
plt.ylabel('Espérance_de_vie')
plt.show()




#la relation entre Espérance_de_vie et VIH/SIDA:

plt.figure(figsize=(10,6))
sns.scatterplot(x='VIH/SIDA', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et VIH/SIDA')
plt.xlabel('VIH/SIDA')
plt.ylabel('Espérance_de_vie')
plt.show()





#la relation entre Espérance_de_vie et PIB:

plt.figure(figsize=(10,6))
sns.scatterplot(x='PIB', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et PIB')
plt.xlabel('PIB')
plt.ylabel('Espérance_de_vie')
plt.show()




#la relation entre Espérance_de_vie et IMC:

plt.figure(figsize=(10,6))
sns.scatterplot(x='IMC', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et IMC')
plt.xlabel('IMC')
plt.ylabel('Espérance_de_vie')
plt.show()



#la relation entre Espérance_de_vie et Scolarité:

plt.figure(figsize=(10,6))
sns.scatterplot(x='Scolarité', y='Espérance_de_vie', data= base)
plt.title('la relation entre Espérance_de_vie et Scolarité')
plt.xlabel('Scolarité')
plt.ylabel('Espérance_de_vie')
plt.show()




#Évolution de l'Espérance de vie au fil des années
       
plt.figure(figsize=(12, 8))
sns.lineplot(x='Année', y='Espérance_de_vie', data=base, ci=None)
plt.title('Évolution de Espérance_de_vie au fil des années')
plt.xlabel('Année')
plt.ylabel('Espérance_de_vie')
plt.show()



         
#visualisation de l'esperence de vie en fct des pays:

df_sorted = base2.sort_values(by='Espérance_de_vie', ascending=True)

plt.figure(figsize=(80, 100))
sns.barplot(x='Espérance_de_vie', y='pays', data=df_sorted, palette='viridis')
plt.title('Espérance de vie par pays')
plt.xlabel('Espérance de vie')
plt.ylabel('pays')
plt.show()






###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
#              Entrainements des modeles de maching learning                                                                       #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################

  



  
#deviser la base de donnees d'entrainement et la base de donnees de test :
    
X = X.drop(['pays'],axis = 1)    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)




#affichage des tailles de chaque base :
    
print("la taille de X_train est : ",X_train.shape)
print("la taille de X_test est : ",X_test.shape)
print("la taille de Y_train est : ",Y_train.shape)
print("la taille de Y_test est : ",Y_test.shape)







#les choix des modeles :

models = {
    'LinearRegression':LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor(),
    'model SVR':SVR(),
    'GradientBoostingRegressor':GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }






#la fonction de précision:

def accuracy_model(Y_test,Y_pred,r = False):
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    if r:
        return mse, r2
    else:
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R2): {r2}")
        print("\n")






#la fonction d'application des modéles:
   
def training(models,X_train,Y_train,X_test,Y_test):
    for nom,model in models.items():
        print(nom,"  :  ")
        print("\n")
        if nom == 'model SVR' or nom == 'RandomForestRegressor':
          #normalisation des donnees:(seulement ces 2 modele a besion de normaliser tt les donnes)
          
            scaler = StandardScaler()
            X_train_1 = scaler.fit_transform(X_train)
            X_test_1 = scaler.transform(X_test)
            model.fit(X_train_1,Y_train)
            accuracy_model(Y_test, model.predict(X_test_1))
        else:
         model.fit(X_train,Y_train)
         accuracy_model(Y_test, model.predict(X_test))
         
         print("#" * 20)
         print("\n")
         
         
         
         



#afficher les precisions de chaque modéle :

training(models, X_train, Y_train, X_test, Y_test)

















###############################################################################
#                                                                             #
#                                                                             #
#                                                                             #
#                           déploiement de modele:                            #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################








#Le choix des variables ayant le plus d'impact sur l'espérance de vie (matrice de corrélation).:

X_2 = X[['Statut','Année','Polio','Diphtérie','VIH/SIDA','PIB','IMC','Scolarité','Alcool','Maigreur_1_19_ans']]
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_2, Y, test_size=0.2, random_state=42)






#affichage des tailles :
    
print("la taille de X_train est : ",X2_train.shape)
print("la taille de X_test est : ",X2_test.shape)
print("la taille de Y_train est : ",Y2_train.shape)
print("la taille de Y_test est : ",Y2_test.shape) 







#appel directement au fct training:

training(models, X2_train, Y2_train, X2_test, Y2_test)








#Appliquer le modéle de Random ForestRegressor sur la base de donnees :
    
model = RandomForestRegressor()
model.fit(X_2,Y)





#Enregestrer le modele sur le fichioer model.pkl :
pickle.dump(model, open("model.pkl","wb"))




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#voir le drive pour consulter mon application web il suffit de lencer le fichier 
#app.py pour consulter directement sans aucun probleme
#aussi il des problemes de la mis a jour de la bibloi sklearn il faut appliquer 
#la dernier version pour que le code soit executer!!!





           









