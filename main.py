import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import streamlit as st


# ========== Функции для обработки данных ==========

def load_and_preprocess_data(fuel_file, oil_file):
    # Загрузка данных
    fuel_data = pd.read_csv(fuel_file)
    oil_data = pd.read_csv(oil_file)

    # Преобразование цен на топливо
    fuel_data['fuel_price'] = fuel_data['fuel_price'].str.replace(',', '.').astype(float)
    fuel_data['date'] = fuel_data['date'].str.replace('"', '')

    # Преобразование цен на нефть
    oil_data['oil_price'] = oil_data['oil_price'].str.replace('.', '')
    oil_data['date'] = oil_data['date'].str.replace('"', '')
    oil_data['oil_price'] = oil_data['oil_price'].str.replace(',', '.').astype(float) / 10

    # Объединение данных
    data = pd.merge(fuel_data, oil_data, on="date")

    # Очистка данных
    for column in ['fuel_price', 'oil_price']:
        if column in data.columns:
            data[column] = data[column].fillna(data[column].mean())

    data.dropna(inplace=True)

    # Преобразование столбцов для модели
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    return data


def plot_data(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='fuel_price', data=data, label="Fuel Price")
    sns.lineplot(x='date', y='oil_price', data=data, label="Oil Price")

    # Устанавливаем заголовок
    plt.title("Цены на топливо и нефть во времени")

    # Настройка оси X
    plt.xticks(rotation=45, ha='right')  # Поворот меток на оси X
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))  # Ограничение на количество меток

    # Добавление легенды
    plt.legend()

    # Показать график
    plt.tight_layout()  # Автоматическая подгонка для лучшего отображения
    return plt


# ========== Функции для обучения модели ==========

def train_models(x_train, y_train):
    # Параметры для GridSearchCV для каждой модели
    param_grid_gb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }

    param_grid_xg = {
        'max_depth': [3, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Модели
    gb_model = GradientBoostingRegressor(random_state=42)
    xg_model = xgb.XGBRegressor(random_state=42)
    rf_model = RandomForestRegressor(random_state=42)

    # GridSearch для каждой модели
    gb_grid = GridSearchCV(gb_model, param_grid_gb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # xg_grid = GridSearchCV(xg_model, param_grid_xg, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Обучение моделей с подбором параметров
    gb_grid.fit(x_train, y_train)
    xg_model.fit(x_train, y_train)
    rf_grid.fit(x_train, y_train)

    # Лучшие параметры и лучшие модели
    print("Лучшие параметры для GradientBoostingRegressor:", gb_grid.best_params_)
    # print("Лучшие параметры для XGBRegressor:", xg_grid.best_params_)
    print("Лучшие параметры для RandomForestRegressor:", rf_grid.best_params_)

    # Возвращаем обученные модели с лучшими параметрами
    return gb_grid.best_estimator_, xg_model, rf_grid.best_estimator_


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse


# ========== Функции для Streamlit ==========

def predict_price(model, oil_price, year, month, day):
    prediction = model.predict([[oil_price, year, month, day]])
    return prediction[0]


def display_results(mae_rf, rmse_rf, mae_gb, rmse_gb, mae_xg, rmse_xg):
    st.write("### Метрики модели RandomForestRegressor")
    st.write(f"MAE: {mae_rf:.2f}")
    st.write(f"RMSE: {rmse_rf:.2f}")

    st.write('### Метрики модели GradientBoostingRegressor')
    st.write(f"MAE: {mae_gb:.2f}")
    st.write(f"RMSE: {rmse_gb:.2f}")

    st.write('### Метрики модели XGBoostRegressor')
    st.write(f"MAE: {mae_xg:.2f}")
    st.write(f"RMSE: {rmse_xg:.2f}")


# ========== Основной блок ==========

def main():
    # Streamlit интерфейс
    st.title("Прогнозирование цен на топливо")

    st.sidebar.header("Входные данные")
    oil_price = st.sidebar.number_input("Введите цену на нефть", value=80.0)
    year = st.sidebar.number_input("Введите год", min_value=2020, max_value=2030, value=2024)
    month = st.sidebar.selectbox("Выберите месяц", range(1, 13))
    day = st.sidebar.selectbox("Выберете день", range(1, 32))
    fuel_proces = st.file_uploader("Цена на топливо", type="csv")
    oil_prices = st.file_uploader("Цена на дизель", type="csv")
    if st.sidebar.button("Прогнозировать"):
        if fuel_proces and oil_prices:
            # Загрузка и обработка данных
            data = load_and_preprocess_data(fuel_proces, oil_prices)

            # Проверка на пустой набор данных
            if data.empty:
                print("Ошибка: Набор данных пуст после обработки. Проверьте входные данные.")
                exit()

            # Подготовка данных для модели
            features = ['oil_price', 'year', 'month', 'day']
            x = data[features]
            y = data['fuel_price']

            # Разделение данных
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Обучение моделей
            gb_model, xg_model, rf_model = train_models(x_train, y_train)

            # Оценка моделей
            mae_rf, rmse_rf = evaluate_model(rf_model, x_test, y_test)
            mae_gb, rmse_gb = evaluate_model(gb_model, x_test, y_test)
            mae_xg, rmse_xg = evaluate_model(xg_model, x_test, y_test)
            st.pyplot(plot_data(data))
            # Проверка входных данных
            try:
                oil_price = float(oil_price)
                year = int(year)
                month = int(month)
                day = int(day)
                if month < 1 or month > 12:
                    raise ValueError("Месяц должен быть в диапазоне от 1 до 12.")
                prediction_rf = predict_price(rf_model, oil_price, year, month, day)
                prediction_gr = predict_price(gb_model, oil_price, year, month, day)
                prediction_xg = predict_price(xg_model, oil_price, year, month, day)
                st.write(f"##Прогнозируемая цена на топливо:")
                st.write(f"RandomForest = {prediction_rf:.2f}  ")
                st.write(f"Gradient Boost = {prediction_gr:.2f}  ")
                st.write(f"XGBoost = {prediction_xg:.2f}  ")
            except ValueError as e:
                st.write(f"Ошибка ввода: {e}")

            # Отображение метрик
        display_results(mae_rf, rmse_rf, mae_gb, rmse_gb, mae_xg, rmse_xg)


if __name__ == "__main__":
    main()
