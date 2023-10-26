import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import exp_viz

def main():
    print('Importing data')
    df_data = pd.read_csv("C:/Users/Bianca Cunha/projeto_final_de_programacao/Data/Credit_card.csv")
    df_label = pd.read_csv("C:/Users/Bianca Cunha/projeto_final_de_programacao/Data/Credit_card_label.csv")
    df = df_data.merge(df_label, on='Ind_ID', how='left')
    print('Dataset shape: ', df.shape)

    print('Preprocessing data...')
    df = df.drop(['Ind_ID', 'Mobile_phone'], axis = 1)

    df['GENDER'].fillna(df['GENDER'].mode()[0],inplace=True)
    df['Car_Owner'].fillna(df['Car_Owner'].mode()[0],inplace=True)
    df['Propert_Owner'].fillna(df['Propert_Owner'].mode()[0],inplace=True)
    df['CHILDREN'].fillna(0,inplace=True)
    df['Annual_income'].fillna(df['Annual_income'].mean(),inplace=True)
    df['Type_Income'].fillna(df['Type_Income'].mode()[0],inplace=True)
    df['EDUCATION'].fillna(df['EDUCATION'].mode()[0],inplace=True)
    df['Marital_status'].fillna(df['Marital_status'].mode()[0],inplace=True)
    df['Housing_type'].fillna(df['Housing_type'].mode()[0],inplace=True)
    df['Birthday_count'].fillna(df['Birthday_count'].mean(),inplace=True)
    df['Employed_days'].fillna(df['Employed_days'].mean(),inplace=True)
    df['Work_Phone'].fillna(df['Work_Phone'].mode()[0],inplace=True)
    df['Phone'].fillna(df['Phone'].mode()[0],inplace=True)
    df['EMAIL_ID'].fillna(df['EMAIL_ID'].mode()[0],inplace=True)
    df['Type_Occupation'].fillna(df['Type_Occupation'].mode()[0],inplace=True)
    df['Family_Members'].fillna(df['Family_Members'].mode()[0],inplace=True)

    df = pd.get_dummies(df, dtype=float)

    # Drop columns
    df = df.drop(['GENDER_F', 'Car_Owner_N', 'Propert_Owner_N'], axis = 1)

    # Rename columns name
    new = {'GENDER_M': 'Gender', 'Car_Owner_Y': 'Car_Owner'}
        
    df.rename(columns=new, inplace=True)

    print('Removing outliers...')
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    X = df.drop(["label"], axis=1)
    y = df["label"]
    feature_names = X.columns

    print('Oversampling...')
    X, y = SMOTE().fit_resample(X, y)

    print('Normalizing data...')
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=feature_names)

    print('Splitting train test sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train = pd.DataFrame(X_train, columns = feature_names)
    X_test = pd.DataFrame(X_test, columns = feature_names)

    print('Tranining model...')
    model = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=18)
    model.fit(X_train, y_train)

    print('Generating predictions for test set...')
    # Make prediction on the testing data
    y_pred = model.predict(X_test)

    # Classification Report
    print(classification_report(y_pred, y_test))

    viz = exp_viz.SHAPViz(model, X_test, 'global')
    viz.generate_explanation_visualizations()

if __name__ == "__main__":
    main()