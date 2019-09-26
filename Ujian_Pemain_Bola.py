import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('data.csv')
dfFifa = df[['Name','Age','Overall','Potential']]

dftarget = dfFifa[dfFifa['Age'] <= 25][dfFifa['Overall'] >= 80][dfFifa['Potential'] >= 80]
labeltarget = 'Potential for Recruitment (Target)'
dftarget['Label'] = labeltarget

dfnontarget = dfFifa.drop(dftarget.index)
labelnontarget = 'Not for Recruitment (Non Target)'
dfnontarget['Label'] = labelnontarget

dfComplete = dftarget.append(dfnontarget, ignore_index = True)

x_train, x_test, y_train, y_test = train_test_split(
    dfComplete[['Age', 'Overall', 'Potential']],
    dfComplete['Label'],
    test_size = .1
)

k = StratifiedKFold(n_splits = 100, random_state=None, shuffle=False)

Data = dfComplete[['Age', 'Overall', 'Potential']].values

for train_index, test_index in k.split(Data, dfComplete['Label']):
    x_train = Data[train_index]
    y_train = dfComplete['Label'][train_index]

def k_value():
    k = round((len(x_train)+len(x_test)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k

print('LR Score:',round(cross_val_score(LogisticRegression(solver = 'lbfgs', multi_class = 'auto'),x_train,y_train).mean() * 100), 'percent')
print('DTC Score:',round(cross_val_score(DecisionTreeClassifier(),x_train,y_train).mean() * 100), 'percent')
print('KNN Score:',round(cross_val_score(KNeighborsClassifier(n_neighbors = k_value()),x_train,y_train).mean() * 100), 'percent')

# Test Results
# LR: 93%
# DTC: 89%
# KNN: 94%
# So I will use KNN

model = KNeighborsClassifier(n_neighbors = k_value())
model.fit(x_train, y_train)

dfSoal = pd.DataFrame(
    np.array([
        ['Andik Vermansyah','Madura United FC',27,87,90,'Indonesia'], 
        ['Awan Setho Raharjo','Bhayangkara FC',22,75,83,'Indonesia'],
        ['Bambang Pamungkas','Persija Jakarta',38,85,75,'Indonesia'],
        ['Cristian Gonzales','PSS Sleman',43,90,85,'Indonesia'],
        ['Egy Maulana Vikri','Lechia Gdańsk',18,88,90,'Indonesia'],
        ['Evan Dimas','Barito Putera',24,85,87,'Indonesia'],
        ['Febri Hariyadi','Persib Bandung',23,77,80,'Indonesia'],
        ['Hansamu Yama Pranata','Persebaya Surabaya',24,82,85,'Indonesia'],
        ['Septian David Maulana','PSIS Semarang',22,83,80,'Indonesia'],
        ['Stefano Lilipaly','Bali United',29,88,86,'Indonesia'] 
    ]),
    columns=['Name', 'Club', 'Age', 'Overall', 'Potential', 'Nationality']
)

dfSoal['Potential'] = model.predict(dfSoal[['Age', 'Overall', 'Potential']])

print(dfSoal)