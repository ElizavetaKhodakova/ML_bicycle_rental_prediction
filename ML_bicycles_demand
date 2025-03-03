import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phik

from phik.report import plot_correlation_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import recall_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from catboost import CatBoostClassifier, Pool, cv
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/obulygin/content/main/SeoulBike/seoul_bike_data.csv', encoding='cp1251')
df.describe()
df.nunique()

# проверяем на наличие дубликатов
df.duplicated().sum()

# выводим инфо о количестве каждого уникального значения для каждого столбца
feature_names = df.columns.tolist()
for column in feature_names:
    print (column)
    print (df[column].value_counts(dropna=False))

# проверяем на наличие пропусков
df.isna().sum()

# переименовываем столбцы
data = df.rename(columns={'Rented Bike Count': 'Rented_bike_count',
                                              'Temperature(°C)': 'Temperature',
                                              'Humidity(%)': 'Humidity',
                                              'Wind speed (m/s)': 'Wind_speed',
                                              'Visibility (10m)': 'Visibility',
                                              'Dew point temperature(°C)': 'Dew_point_temperature',
                                              'Solar Radiation (MJ/m2)': 'Solar_radiation',
                                              'Snowfall (cm)': 'Snowfall',
                                              'Functioning Day': 'Functioning_day',
                                              'Rainfall(mm)': 'Rainfall'
                                              })

# меняем формат столбца дата, чтобы взять оттуда инфо для двух новых столбцов
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['day_of_week'] = data['Date'].dt.day_name()
data['month'] = data['Date'].dt.month_name()

data = data.drop(['Date'], axis=1)
plt.figure(figsize=(12, 5))
plot = sns.boxplot(data=data)
plot.set_xticklabels(plot.get_xticklabels(),rotation=90)
plt.show()

data.hist(figsize=(15,10))

sns.pairplot(data)

corr_with_target = data.phik_matrix()['Rented_bike_count'].abs().sort_values()
corr_with_target = corr_with_target.drop("Rented_bike_count")
plt.figure(figsize=(15,5))
plt.bar(corr_with_target.index, corr_with_target.values)
plt.title("Коореляция с целевой переменной (фи-катое)")
plt.xlabel("Признаки")
plt.ylabel('Коэффициент корреляции')
plt.xticks(rotation=70)
plt.show()

phik_overview = data.phik_matrix()
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, figsize=(15, 5))

# делаем кодирование категориальных данных
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Rented_bike_count'], axis=1), data['Rented_bike_count'],  random_state=42)
preprocessor = make_column_transformer(
    (OneHotEncoder(drop = 'first'), make_column_selector(dtype_include=[object, 'category']))) #заменить на все столбцы с делением на тип данных
lr_pipeline = make_pipeline(preprocessor, StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False), ElasticNet(random_state=42))
lr_pipeline.fit(X_train, y_train)
# Извлечение предварительного процессора из пайплайна
preprocessor_step = lr_pipeline.named_steps['columntransformer']
# Преобразование данных с помощью предварительного процессора
transformed_data = pd.DataFrame(preprocessor_step.transform(X_train))
# Преобразование результата в DataFrame для удобства просмотра
transformed_df = pd.DataFrame(transformed_data, columns=preprocessor_step.get_feature_names_out())
# Вывод преобразованных данных
transformed_df
# # объединяем датафреймы (исходный и с закодированными данными)
X_train_encoded = pd.concat([X_train.reset_index(drop=True), transformed_df.reset_index(drop=True)], axis=1)

# строим простую модель
kf = KFold(n_splits=5)
cv_metrics = cross_validate(
    estimator=lr_pipeline,
    X=X_train,
    y=y_train,
    cv=kf,
    scoring='r2',
    return_train_score=True
)
n = X_train.shape[0]
k = X_train.shape[1]
adjusted_r2 = 1 - (1 - cv_metrics['train_score'].mean()) * (n - 1) / (n - k - 1)
print(f"Adj R^2: {adjusted_r2:.4f}")
print(f"Среднее качество на тренировочной выборке: {cv_metrics['train_score'].mean():.4f}")
print(f"Среднее качество на валидационной выборке: {cv_metrics['test_score'].mean():.4f}")

kf = KFold(n_splits=5)
param_grid = {
    'elasticnet__alpha': np.linspace(0, 2000, 10),
    'elasticnet__l1_ratio': np.linspace(0, 1, 10)
}
halving_search = HalvingGridSearchCV(lr_pipeline,
                                     param_grid=param_grid,
                                     scoring='r2',
                                     n_jobs=-1,
                                     cv=KFold(3, shuffle=True, random_state=42),
                                     random_state=42)

halving_search.fit(X_train, y_train)
n = X_train.shape[0]
k = X_train.shape[1]
adjusted_r2 = 1 - (1 - halving_search.best_score_) * (n - 1) / (n - k - 1)
print(f"Adj R^2: {adjusted_r2:.4f}")
print(f"Наилучшее значение R2 при кросс-валидации: {halving_search.best_score_}")
print(f"Наилучшие значения параметров: {halving_search.best_params_}")


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Rented_bike_count', 'Rainfall'], axis=1), data['Rented_bike_count'],  random_state=42)
preprocessor = make_column_transformer(
    (OneHotEncoder(drop = 'first'), ['Seasons', 'Holiday', 'Functioning_day', 'day_of_week', 'month']))
lr_pipeline = make_pipeline(preprocessor, StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False), ElasticNet(random_state=42))
lr_pipeline.fit(X_train, y_train)
kf = KFold(n_splits=5)
param_grid = {
    'elasticnet__alpha': np.linspace(0, 2000, 10),
    'elasticnet__l1_ratio': np.linspace(0, 1, 10)
}
halving_search = HalvingGridSearchCV(lr_pipeline,
                                     param_grid=param_grid,
                                     scoring='r2',
                                     n_jobs=-1,
                                     cv=KFold(3, shuffle=True, random_state=42),
                                     random_state=42)
halving_search.fit(X_train, y_train)
n = X_train.shape[0]
k = X_train.shape[1]
adjusted_r2 = 1 - (1 - halving_search.best_score_) * (n - 1) / (n - k - 1)
print(f"Adj R^2: {adjusted_r2:.4f}")
print(f"Наилучшее значение R2 при кросс-валидации: {halving_search.best_score_}")
print(f"Наилучшие значения параметров: {halving_search.best_params_}")


X_train, X_test, y_train, y_test = train_test_split(data.drop(['Rented_bike_count', 'Rainfall', 'day_of_week', 'Holiday', "Snowfall", 'Wind_speed', "Visibility" ], axis=1), data['Rented_bike_count'],  random_state=42)
preprocessor = make_column_transformer(
    (OneHotEncoder(drop = 'first'), ['Seasons', 'Functioning_day', 'month']))
lr_pipeline = make_pipeline(preprocessor, StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False), ElasticNet(random_state=42))
lr_pipeline.fit(X_train, y_train)
kf = KFold(n_splits=5)
param_grid = {
    'elasticnet__alpha': np.linspace(0, 2000, 10),
    'elasticnet__l1_ratio': np.linspace(0, 1, 10)
}
halving_search = HalvingGridSearchCV(lr_pipeline,
                                     param_grid=param_grid,
                                     scoring='r2',
                                     n_jobs=-1,
                                     cv=KFold(3, shuffle=True, random_state=42),
                                     random_state=42)
halving_search.fit(X_train, y_train)
n = X_train.shape[0]
k = X_train.shape[1]
adjusted_r2 = 1 - (1 - halving_search.best_score_) * (n - 1) / (n - k - 1)
print(f"Adj R^2: {adjusted_r2:.4f}")
print(f"Наилучшее значение R2 при кросс-валидации: {halving_search.best_score_}")
print(f"Наилучшие значения параметров: {halving_search.best_params_}")

cat_features = ['Seasons',   'Functioning_day', 'month', 'Holiday', 'day_of_week']
numeric_features = ['Hour', 'Temperature', 'Humidity', 'Wind_speed', 'Visibility', 'Snowfall', 'Rainfall', 'Dew_point_temperature', 'Solar_radiation']

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Rented_bike_count'], axis=1), data['Rented_bike_count'],  random_state=42)

# Создаем объект Pool для CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)

# Создаем и устанавливаем параметры CatBoost с указанием функции потерь и метрик
params = {
    'random_seed': 42,
    'cat_features': cat_features,
    'custom_metric': 'R2',  
    'loss_function': 'RMSE'  
}

# Обучение модели с использованием кросс-валидации
cv_results = cv(
    train_pool,
    params,
    nfold=5,
    shuffle=True,
    as_pandas=True,  
    early_stopping_rounds=100  
)

# Получение и вывод лучших результатов
best_result = cv_results['test-R2-mean'].max()
print(f"Лучшее значение точности на кросс-валидации: {best_result:.4f}")

# Обучаем модель
cbr = CatBoostRegressor(**params)
cbr.fit(train_pool)

# Получаем важность признаков
feature_importances = cbr.get_feature_importance(Pool(X_train, y_train, cat_features=cat_features))

# Создаем DataFrame для удобства отображения
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Сортируем по важности
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df

cbr_pred = cbr.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
r2 = r2_score(y_test,cbr_pred)
print(r2)

n = X_test.shape[0]
k = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print(adjusted_r2)

# Создаем объект Pool для CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
# Создаем объект CatBoostClassifier
cbr_1 = CatBoostRegressor(
    random_seed=42,
    loss_function='RMSE',
    custom_metric="R2",
)
# Определяем параметры для GridSearch
grid = {
    'depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1],
    'iterations': [100, 1000]
}
# Выполнение Grid Search
grid_search_result = cbr_1.grid_search(
    grid,
    train_pool,
    cv=5,  
    stratified=True,
    shuffle=True,
    verbose=1  
)
# Выводим результаты лучшей модели
print(f"Лучшие параметры: {grid_search_result['params']}")
best_result = cv_results['test-RMSE-mean'].max()
print(f"Лучшее значение точности на кросс-валидации: {best_result:.4f}")

cbr_1.fit(train_pool)
cbr_1_pred = cbr_1.predict(X_test)
r2 = r2_score(y_test,cbr_1_pred)
print(r2)

n = X_test.shape[0]
k = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print(adjusted_r2)

