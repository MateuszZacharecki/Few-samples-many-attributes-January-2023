import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import RFE

if __name__ == '__main__':
    Training_1 = pd.read_csv("1_anthracyclineTaxaneChemotherapy_training.csv")
    Training_2 = pd.read_csv("10_skinPsoriatic_training.csv")
    Training_3 = pd.read_csv("2_brainTumour_training.csv")
    Training_4 = pd.read_csv("3_BurkittLymphoma_training.csv")
    Training_5 = pd.read_csv("4_gingivalPeriodontits_training.csv")
    Training_6 = pd.read_csv("5_heartFailurFactors_training.csv")
    Training_7 = pd.read_csv("6_hepatitisC_training.csv")
    Training_8 = pd.read_csv("7_humanGlioma_training.csv")
    Training_9 = pd.read_csv("8_ovarianTumour_training.csv")
    Training_10 = pd.read_csv("9_septicShock_training.csv")

    # Training_1:

    print('Training set 1:')
    print(Training_1)
    print()

    Training = Training_1
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'pCR', 'cel'] = 1

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_1 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_1[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_2:

    print('Training set 2:')
    print(Training_2)
    print()

    Training = Training_2
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'normal', 'cel'] = 1
    Training.loc[Training['target'] == 'involved', 'cel'] = 2

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_2 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_2[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_3:

    print('Training set 3:')
    print(Training_3)
    print()

    Training = Training_3
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'epilepsy', 'cel'] = 1
    Training.loc[Training['target'] == 'glioblastoma', 'cel'] = 2
    Training.loc[Training['target'] == 'astrocytoma', 'cel'] = 3

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_3 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_3[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_4:

    print('Training set 4:')
    print(Training_4)
    print()

    Training = Training_4
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == '_Molecular.Diagnosis_:_intermediate', 'cel'] = 1
    Training.loc[Training['target'] == '_Molecular.Diagnosis_:_mBL', 'cel'] = 2

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_4 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_4[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_5:

    print('Training set 5:')
    print(Training_5)
    print()

    Training = Training_5
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'Diseased', 'cel'] = 1

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_5 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_5[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_6:

    print('Training set 6:')
    print(Training_6)
    print()

    Training = Training_6
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'ischemic_cardiomyopathy', 'cel'] = 1
    Training.loc[Training['target'] == 'idiopathic_dilated_cardiomyopathy', 'cel'] = 2

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_6 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_6[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_7:

    print('Training set 7:')
    print(Training_7)
    print()

    Training = Training_7
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'Tissue:_HCC', 'cel'] = 1
    Training.loc[Training['target'] == 'Tissue:_cirrhosisHCC', 'cel'] = 2
    Training.loc[Training['target'] == 'Tissue:_cirrhosis', 'cel'] = 3

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_7 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_7[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_8:

    print('Training set 8:')
    print(Training_8)
    print()

    Training = Training_8
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'primaryglioblastoma', 'cel'] = 1
    Training.loc[Training['target'] == 'nontumortissue', 'cel'] = 2
    Training.loc[Training['target'] == 'anaplasticmixedglioma', 'cel'] = 3
    Training.loc[Training['target'] == 'secondaryglioblastoma', 'cel'] = 4

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_8 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_8[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_9:

    print('Training set 9:')
    print(Training_9)
    print()

    Training = Training_9
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'Malignant:_Ser/PapSer', 'cel'] = 1
    Training.loc[Training['target'] == 'Malignant:_Endo', 'cel'] = 2

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_9 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_9[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    # Training_10:

    print('Training set 10:')
    print(Training_10)
    print()

    Training = Training_10
    print('Factorization of data from column target:')
    print(Training['target'].value_counts())
    print()

    Training['cel'] = 0
    Training.loc[Training['target'] == 'Septic_Shock_day3', 'cel'] = 1
    Training.loc[Training['target'] == 'Sepsis_day3', 'cel'] = 2
    Training.loc[Training['target'] == 'SIRS_day3', 'cel'] = 3
    Training.loc[Training['target'] == 'Control_day3', 'cel'] = 4

    Training_label = Training['cel'].to_numpy(copy=True)

    Training = Training.drop(columns=['cel', 'target'])

    Training_norm = MinMaxScaler().fit_transform(Training)
    chi_selector = SelectKBest(chi2, k=800)  # biorę 800 najlepszych kolumn z danych
    chi_selector.fit(Training_norm, Training_label)
    chi_support = chi_selector.get_support()
    chi_feature = Training.loc[:, chi_support].columns.tolist()

    Training_chi = Training[chi_feature]

    Training_norm_chi = MinMaxScaler().fit_transform(Training_chi)
    rfe = RFE(svm.SVC(kernel='linear', C=1.0),
              n_features_to_select=80)
    rfe.fit(Training_norm_chi, Training_label)
    rfe_support = rfe.get_support()
    rfe_feature = Training_chi.loc[:, rfe_support].columns.tolist()

    Training_rfe = Training[rfe_feature]
    cols_index_10 = [Training.columns.get_loc(col) + 1 for col in rfe_feature]

    A = np.arange(80)
    for i in range(80):
        A[i] = cols_index_10[i]

    print('Filtered data:')
    print(A)
    print()

    Training_rfe_norm = MinMaxScaler().fit_transform(Training_rfe)

    X_train, X_test, y_train, y_test = train_test_split(Training_rfe_norm, Training_label, test_size=0.25)

    clf = svm.SVC(kernel='linear', C=1.0)  # Linear Kernel

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print('Prediction:')
    print(bal_acc)
    print()

    lista = [cols_index_1, cols_index_2, cols_index_3, cols_index_4, cols_index_5,
             cols_index_6, cols_index_7, cols_index_8, cols_index_9, cols_index_10]

    k = 1
    for i in lista:
        print(f'Selected columns from set {k}:')
        print(i)
        print()
        k = k + 1
