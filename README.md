# Анализ факторов, влияющих на спрос велосипедов

Работа с [данными](https://raw.githubusercontent.com/obulygin/content/main/SeoulBike/seoul_bike_data.csv) сервиса проката велосипедов в Корее за год.

## Цель работы
Цель данного проекта — изучить данные о прокате велосипедов и выявить факторы, влияющие на спрос на велосипеды, а также построить модель с adjusted R2 не менее 89%.

## Задачи
В процессе работы над проектом выполнены следующие задачи:

1. **Предобработка данных**
   - Проверка данных на наличие выбросов, ошибочных значений, пропусков, дубликатов и некорректных типов.
   - Очистка и подготовка данных для дальнейшего анализа.

2. **Исследовательский анализ данных (EDA)**
   - Проведение анализа с использованием визуализации.
   - Изучение распределений и взаимосвязей между признаками для выявления паттернов, влияющих на спрос.

3. **Подготовка данных для построения модели**
   - Кодирование категориальных признаков.
   - Масштабирование числовых признаков.
   - Разбиение выборки на обучающую и тестовую.

4. **Построение регрессионной модели**
   - Реализация базовой регрессионной модели для прогнозирования количества велосипедов, взятых в прокат.
   - Использование модели ElasticNet для повышения качества прогнозирования.

5. **Оптимизация модели**
   - Использование методов выбора признаков (Feature Selection) и подбора гиперпараметров для улучшения модели.
   - Подбор гиперпараметров с помощью метода Halving Grid Search.
   - Определение наиболее значимых признаков, влияющих на спрос.
   - Фактически, полученный **лучший результат** составил **94%**.

## Используемые библиотеки
В проекте были использованы следующие библиотеки:
- `numpy`
- `pandas`
- `seaborn`
- `phik`
- `scikit-learn`
- `catboost`
- `imblearn`
- `matplotlib`


# Analysis of Factors Influencing Bike Demand

Working with [data](https://raw.githubusercontent.com/obulygin/content/main/SeoulBike/seoul_bike_data.csv) from the bike rental service in South Korea over the course of a year.

## Project Goal
The goal of this project is to analyze bike rental data, identify the factors that influence bike demand and also build a model with adjusted R2 of at least 89%.

## Tasks
In the process of working on this project, the following tasks were completed:

1. **Data Preprocessing**
   - Checking for outliers, erroneous values, missing data, duplicates, and incorrect types.
   - Cleaning and preparing the data for further analysis.

2. **Exploratory Data Analysis (EDA)**
   - Conducting analysis using visualization techniques.
   - Examining distributions and relationships between features to identify patterns influencing demand.

3. **Data Preparation for Model Building**
   - Encoding categorical features.
   - Scaling numerical features.
   - Splitting the dataset into training and testing sets.

4. **Building a Regression Model**
   - Implementing a baseline regression model to predict the number of bikes rented.
   - Using the ElasticNet model to improve prediction quality.

5. **Model Optimization**
   - Applying feature selection methods and hyperparameter tuning to enhance the model.
   - Selecting hyperparameters using the Halving Grid Search method.
   - Identifying the most significant features influencing demand.
   - The **best achieved result** was **94%**.

## Libraries Used
The following libraries were used in this project:
- `numpy`
- `pandas`
- `seaborn`
- `phik`
- `scikit-learn`
- `catboost`
- `imblearn`
- `matplotlib`
