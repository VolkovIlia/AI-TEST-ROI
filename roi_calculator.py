import pandas as pd
import argparse
import sys
import os
import numpy as np # Добавим numpy для обработки inf и случайных чисел
from tqdm import tqdm # Для индикатора прогресса симуляции
import matplotlib
matplotlib.use('Agg') # Используем бэкенд Agg для сохранения файлов без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter # Для форматирования осей
from scipy.stats import spearmanr # Для анализа чувствительности
import warnings

# --- Подавление FutureWarnings от seaborn/matplotlib ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Настройка стилей графиков ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Размер графиков по умолчанию
# Попробуем установить русскую локаль для форматирования чисел, если возможно
try:
    import locale
    # locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8') # Закомментировано
    # plt.rcParams['axes.formatter.use_locale'] = True
    # print("Русская локаль для графиков установлена.")
except ImportError:
    print("Модуль locale не найден, используется стандартное форматирование чисел.")
except Exception as e:
    print(f"Не удалось установить русскую локаль ({e}), используется стандартное форматирование чисел.")
    # plt.rcParams['axes.formatter.use_locale'] = False

plt.rcParams['figure.max_open_warning'] = False # Отключить предупреждение о большом кол-ве фигур

# --- Константы для имен столбцов ---

# Входные данные (из CSV)
COL_PERIOD = "Период"
COL_PERIOD_DAYS = "Период_Отчетности_Дни"
COL_TOTAL_REQUESTS = "Общее_Число_Запросов"
COL_TOTAL_PROC_TIME_HOURS = "Суммарное_Время_Обработки_Всех_Запросов_Часы"
COL_AVG_PROC_TIME_BEFORE_AI_HOURS = "Среднее_Время_Обработки_До_ИИ_Часы"
COL_TOTAL_OPERATING_COSTS_RUB = "Общие_Операционные_Затраты_На_Обработку_Руб"
COL_CSAT_GOOD_EXCELLENT = "Число_Оценок_Хорошо_Отлично_CSAT"
COL_CSAT_TOTAL_RATINGS = "Общее_Число_Оценок_CSAT"
COL_NPS_PROMOTERS = "Число_Промоутеров_NPS"
COL_NPS_DETRACTORS = "Число_Детракторов_NPS"
COL_NPS_TOTAL_RESPONDENTS = "Общее_Число_Опрошенных_NPS"
COL_FCR_RESOLVED_FIRST_TIME = "Число_Решенных_С_Первого_Раза"
COL_COMPLAINTS = "Число_Жалоб"
COL_REPEAT_REQUESTS = "Число_Повторных_Обращений"
COL_RESOLVED_ON_TIME = "Число_Решенных_В_Срок"
COL_ERRORS_CORRECTIONS = "Число_Ошибок_Корректировок"
COL_HIGH_PRIORITY_REQUESTS = "Число_Обращений_Высокий_Приоритет"
COL_HIGH_PRIORITY_REACTION_TIME_HOURS = "Суммарное_Время_Реакции_Высокий_Приоритет_Часы"
COL_LOW_PRIORITY_REQUESTS = "Число_Обращений_Низкий_Приоритет"
COL_LOW_PRIORITY_RESOLUTION_TIME_HOURS = "Суммарное_Время_Решения_Низкий_Приоритет_Часы"
COL_REQUESTS_WITH_FORECAST = "Число_Обращений_С_Прогнозом"
COL_FORECAST_ABS_ERROR_HOURS = "Суммарная_Абсолютная_Ошибка_Прогноза_Часы"
COL_RESOLVED_ON_AI_TIME = "Число_Решенных_В_Срок_ИИ"
COL_STATUS_REQUESTS = "Число_Запросов_Статус"
COL_TOTAL_TEXT_REQUESTS = "Общее_Число_Текстовых_Запросов"
COL_TEXT_PROCESSED_BY_AI = "Число_Текстовых_Обработано_ИИ"
COL_TEXT_INTERPRETED_CORRECTLY = "Число_Текстовых_Правильно_Интерпретировано"
COL_TOTAL_TEXT_PROC_TIME_HOURS = "Суммарное_Время_Обработки_Текстовых_Часы"
COL_AVG_TEXT_PROC_MANUAL_HOURS = "Среднее_Время_Обработки_Текстовых_Вручную_Часы"
COL_SATISFACTION_INDEX_PCT = "Индекс_Удовлетворенности_Госуслугами_Процент"
COL_TRUST_INDEX_PCT = "Уровень_Доверия_ИИ_Процент"
COL_ONLINE_REQUESTS = "Число_Онлайн_Запросов"
COL_OFFLINE_REQUESTS = "Число_Офлайн_Запросов"
COL_POSITIVE_REVIEWS = "Число_Положительных_Отзывов"
COL_NEGATIVE_REVIEWS = "Число_Негативных_Отзывов"
COL_COST_PER_HOUR_RUB = "Стоимость_Часа_Сотрудника_Руб"
COL_PROJECT_COSTS_RUB = "Затраты_На_Проект_За_Период_Руб"
COL_DISCOUNT_RATE_ANNUAL_PCT = "Ставка_Дисконтирования_Годовая_%"

# Рассчитанные метрики (префиксы для ясности)
METRIC_AVG_PROC_TIME_HOURS = "Метрика_Среднее_Время_Обработки_Часы"
METRIC_THROUGHPUT_PER_DAY = "Метрика_Пропускная_Способность_В_День"
METRIC_COST_PER_REQUEST_RUB = "Метрика_Стоимость_Обработки_Запроса_Руб"
METRIC_HOURS_SAVED = "Метрика_Сэкономлено_Часов_Сотрудников"
METRIC_CSAT_PCT = "Метрика_CSAT_%"
METRIC_NPS = "Метрика_NPS"
METRIC_FCR_PCT = "Метрика_FCR_%"
METRIC_ON_TIME_PCT = "Метрика_Доля_Решенных_В_Срок_%"
METRIC_ERROR_RATE_PCT = "Метрика_Уровень_Ошибок_%"
METRIC_AVG_HIGH_PRIORITY_REACTION_TIME_HOURS = "Метрика_Среднее_Время_Реакции_Выс_Приоритет_Часы"
METRIC_PRIORITIZATION_RATIO = "Метрика_Коэффициент_Приоритизации"
METRIC_FORECAST_MAE_HOURS = "Метрика_MAE_Прогноза_Часы"
METRIC_ON_AI_TARGET_PCT = "Метрика_Доля_Соблюдения_Срока_ИИ_%"
METRIC_TEXT_AUTOMATION_RATE_PCT = "Метрика_Доля_Автоматизации_Текст_%"
METRIC_TEXT_ACCURACY_PCT = "Метрика_Точность_Распознавания_Текст_%"
METRIC_AVG_TEXT_PROC_TIME_HOURS = "Метрика_Среднее_Время_Обработки_Текст_Часы"
METRIC_ONLINE_ENGAGEMENT_PCT = "Метрика_Доля_Онлайн_Обращений_%"

# Рассчитанные эффекты (префиксы)
EFFECT_FOT_SAVINGS_RUB = "Эффект_Экономия_ФОТ_Руб"
EFFECT_TEXT_PROC_SAVINGS_RUB = "Эффект_Экономия_На_Обработке_Текста_Руб"
# Список всех столбцов с денежными эффектами для суммирования
ECONOMIC_EFFECT_COLUMNS = [
    EFFECT_FOT_SAVINGS_RUB,
    EFFECT_TEXT_PROC_SAVINGS_RUB
    # Добавить сюда другие столбцы с эффектами, если они появятся
]

# Итоговые показатели
CALC_TOTAL_BENEFIT_RUB = "Расчет_Совокупная_Выгода_Руб"
CALC_ROI_PCT = "Расчет_ROI_%"
# Новые расчетные показатели для NPV
CALC_PERIOD_YEAR = "Расчет_Период_Год"
CALC_DISCOUNT_FACTOR = "Расчет_Фактор_Дисконтирования"
CALC_DISCOUNTED_BENEFIT_RUB = "Расчет_Дисконтированная_Выгода_Руб"
CALC_DISCOUNTED_COST_RUB = "Расчет_Дисконтированные_Затраты_Руб"
CALC_DISCOUNTED_CASH_FLOW_RUB = "Расчет_Дисконтированный_Денежный_Поток_Руб"
CALC_NPV_RUB = "Расчет_NPV_Накопленный_Руб"
CALC_DISCOUNTED_ROI_PCT = "Расчет_Дисконтированный_ROI_%"


def safe_divide_np(numerator, denominator, default=0.0):
    """
    Безопасное деление для numpy массивов.
    Возвращает default при делении на ноль, NaN или бесконечность.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = default # Заменяем inf, -inf, NaN
        # Также обрабатываем случаи, когда знаменатель был 0 или NaN изначально
        if np.isscalar(denominator):
             if denominator == 0 or np.isnan(denominator):
                 result.fill(default)
        else:
            result[denominator == 0] = default
            result[np.isnan(denominator)] = default
        if np.isscalar(numerator):
            if np.isnan(numerator):
                result.fill(default)
        else:
            result[np.isnan(numerator)] = default
    return result


def load_data(csv_path):
    """Загружает данные из CSV файла и выполняет базовую проверку."""
    if not os.path.exists(csv_path):
        print(f"Ошибка: Файл не найден: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
        print(f"Данные успешно загружены из {csv_path}")

        # Преобразование числовых столбцов, заменяя ошибки на NaN
        numeric_cols = [
            COL_PERIOD_DAYS, COL_TOTAL_REQUESTS, COL_TOTAL_PROC_TIME_HOURS,
            COL_AVG_PROC_TIME_BEFORE_AI_HOURS, COL_TOTAL_OPERATING_COSTS_RUB,
            COL_CSAT_GOOD_EXCELLENT, COL_CSAT_TOTAL_RATINGS, COL_NPS_PROMOTERS,
            COL_NPS_DETRACTORS, COL_NPS_TOTAL_RESPONDENTS, COL_FCR_RESOLVED_FIRST_TIME,
            COL_COMPLAINTS, COL_REPEAT_REQUESTS, COL_RESOLVED_ON_TIME,
            COL_ERRORS_CORRECTIONS, COL_HIGH_PRIORITY_REQUESTS,
            COL_HIGH_PRIORITY_REACTION_TIME_HOURS, COL_LOW_PRIORITY_REQUESTS,
            COL_LOW_PRIORITY_RESOLUTION_TIME_HOURS, COL_REQUESTS_WITH_FORECAST,
            COL_FORECAST_ABS_ERROR_HOURS, COL_RESOLVED_ON_AI_TIME, COL_STATUS_REQUESTS,
            COL_TOTAL_TEXT_REQUESTS, COL_TEXT_PROCESSED_BY_AI,
            COL_TEXT_INTERPRETED_CORRECTLY, COL_TOTAL_TEXT_PROC_TIME_HOURS,
            COL_AVG_TEXT_PROC_MANUAL_HOURS, COL_SATISFACTION_INDEX_PCT,
            COL_TRUST_INDEX_PCT, COL_ONLINE_REQUESTS, COL_OFFLINE_REQUESTS,
            COL_POSITIVE_REVIEWS, COL_NEGATIVE_REVIEWS, COL_COST_PER_HOUR_RUB,
            COL_PROJECT_COSTS_RUB, COL_DISCOUNT_RATE_ANNUAL_PCT # Добавили новый столбец
        ]
        # Проверяем наличие столбцов перед преобразованием
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        if missing_cols:
            print(f"Предупреждение: Следующие ожидаемые столбцы не найдены в CSV и будут заполнены нулями: {', '.join(missing_cols)}")
            for col in missing_cols:
                df[col] = 0 # Создаем столбец с нулями

        for col in numeric_cols:
             # Пропускаем столбцы, которых не было изначально (уже созданы выше)
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')


        # Заполнение NaN нулями для простоты расчетов (можно изменить логику при необходимости)
        # Делаем это после преобразования всех колонок
        df.fillna(0, inplace=True)
        print("Числовые столбцы преобразованы, пропуски заполнены нулями.")

        return df
    except Exception as e:
        print(f"Ошибка при чтении или обработке CSV файла {csv_path}: {e}")
        sys.exit(1)

# --- Функции расчета метрик (для DataFrame) ---
def calculate_metrics_df(df):
    """Рассчитывает все метрики и добавляет их в DataFrame (для детерминированного расчета)."""
    # Оптимизация процессов
    df[METRIC_AVG_PROC_TIME_HOURS] = safe_divide_np(df[COL_TOTAL_PROC_TIME_HOURS], df[COL_TOTAL_REQUESTS])
    df[METRIC_THROUGHPUT_PER_DAY] = safe_divide_np(df[COL_TOTAL_REQUESTS], df[COL_PERIOD_DAYS])
    df[METRIC_COST_PER_REQUEST_RUB] = safe_divide_np(df[COL_TOTAL_OPERATING_COSTS_RUB], df[COL_TOTAL_REQUESTS])
    df[METRIC_HOURS_SAVED] = (df[COL_AVG_PROC_TIME_BEFORE_AI_HOURS] - df[METRIC_AVG_PROC_TIME_HOURS]) * df[COL_TOTAL_REQUESTS]
    df[METRIC_HOURS_SAVED] = df[METRIC_HOURS_SAVED].clip(lower=0)

    # Качество обслуживания
    df[METRIC_CSAT_PCT] = safe_divide_np(df[COL_CSAT_GOOD_EXCELLENT] * 100, df[COL_CSAT_TOTAL_RATINGS])
    df[METRIC_NPS] = safe_divide_np((df[COL_NPS_PROMOTERS] - df[COL_NPS_DETRACTORS]) * 100, df[COL_NPS_TOTAL_RESPONDENTS])
    df[METRIC_FCR_PCT] = safe_divide_np(df[COL_FCR_RESOLVED_FIRST_TIME] * 100, df[COL_TOTAL_REQUESTS])

    # Своевременность и точность
    df[METRIC_ON_TIME_PCT] = safe_divide_np(df[COL_RESOLVED_ON_TIME] * 100, df[COL_TOTAL_REQUESTS])
    df[METRIC_ERROR_RATE_PCT] = safe_divide_np(df[COL_ERRORS_CORRECTIONS] * 100, df[COL_TOTAL_REQUESTS])

    # Приоритизация
    df[METRIC_AVG_HIGH_PRIORITY_REACTION_TIME_HOURS] = safe_divide_np(df[COL_HIGH_PRIORITY_REACTION_TIME_HOURS], df[COL_HIGH_PRIORITY_REQUESTS])
    avg_low_priority_time = safe_divide_np(df[COL_LOW_PRIORITY_RESOLUTION_TIME_HOURS], df[COL_LOW_PRIORITY_REQUESTS])
    df[METRIC_PRIORITIZATION_RATIO] = safe_divide_np(avg_low_priority_time, df[METRIC_AVG_HIGH_PRIORITY_REACTION_TIME_HOURS])

    # Подсказки по срокам
    df[METRIC_FORECAST_MAE_HOURS] = safe_divide_np(df[COL_FORECAST_ABS_ERROR_HOURS], df[COL_REQUESTS_WITH_FORECAST])
    df[METRIC_ON_AI_TARGET_PCT] = safe_divide_np(df[COL_RESOLVED_ON_AI_TIME] * 100, df[COL_REQUESTS_WITH_FORECAST])

    # Обработка текстовых запросов
    df[METRIC_TEXT_AUTOMATION_RATE_PCT] = safe_divide_np(df[COL_TEXT_PROCESSED_BY_AI] * 100, df[COL_TOTAL_TEXT_REQUESTS])
    df[METRIC_TEXT_ACCURACY_PCT] = safe_divide_np(df[COL_TEXT_INTERPRETED_CORRECTLY] * 100, df[COL_TEXT_PROCESSED_BY_AI])
    df[METRIC_AVG_TEXT_PROC_TIME_HOURS] = safe_divide_np(df[COL_TOTAL_TEXT_PROC_TIME_HOURS], df[COL_TOTAL_TEXT_REQUESTS])

    # Удовлетворенность и доверие
    total_online_offline = df[COL_ONLINE_REQUESTS] + df[COL_OFFLINE_REQUESTS]
    df[METRIC_ONLINE_ENGAGEMENT_PCT] = safe_divide_np(df[COL_ONLINE_REQUESTS] * 100, total_online_offline)

    return df

# --- Функции расчета денежных эффектов (для DataFrame) ---
def calculate_economic_effects_df(df):
    """Рассчитывает денежные эффекты и добавляет их в DataFrame (для детерминированного расчета)."""
    df[EFFECT_FOT_SAVINGS_RUB] = df[METRIC_HOURS_SAVED] * df[COL_COST_PER_HOUR_RUB]
    df[EFFECT_TEXT_PROC_SAVINGS_RUB] = df[COL_TEXT_PROCESSED_BY_AI] * df[COL_AVG_TEXT_PROC_MANUAL_HOURS] * df[COL_COST_PER_HOUR_RUB]
    return df

# --- Функция расчета ROI и NPV (для DataFrame) ---
def calculate_roi_npv_df(df):
    """Рассчитывает совокупную выгоду, ROI, NPV и дисконтированный ROI (для детерминированного расчета)."""
    for col in ECONOMIC_EFFECT_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    existing_effect_cols = [col for col in ECONOMIC_EFFECT_COLUMNS if col in df.columns]
    if not existing_effect_cols:
         df[CALC_TOTAL_BENEFIT_RUB] = 0
    else:
        df[CALC_TOTAL_BENEFIT_RUB] = df[existing_effect_cols].sum(axis=1)

    df[CALC_ROI_PCT] = safe_divide_np(
        (df[CALC_TOTAL_BENEFIT_RUB] - df[COL_PROJECT_COSTS_RUB]),
        df[COL_PROJECT_COSTS_RUB],
        default=0.0
    ) * 100

    df[CALC_PERIOD_YEAR] = df.index * 0.5
    discount_rate = df[COL_DISCOUNT_RATE_ANNUAL_PCT] / 100.0
    df[CALC_DISCOUNT_FACTOR] = 1 / (1 + discount_rate)**df[CALC_PERIOD_YEAR]

    df[CALC_DISCOUNTED_BENEFIT_RUB] = df[CALC_TOTAL_BENEFIT_RUB] * df[CALC_DISCOUNT_FACTOR]
    df[CALC_DISCOUNTED_COST_RUB] = df[COL_PROJECT_COSTS_RUB] * df[CALC_DISCOUNT_FACTOR]
    df[CALC_DISCOUNTED_CASH_FLOW_RUB] = df[CALC_DISCOUNTED_BENEFIT_RUB] - df[CALC_DISCOUNTED_COST_RUB]

    df[CALC_NPV_RUB] = df[CALC_DISCOUNTED_CASH_FLOW_RUB].cumsum()

    cumulative_discounted_costs = df[CALC_DISCOUNTED_COST_RUB].cumsum()
    # Используем where в np.divide для безопасного деления
    df[CALC_DISCOUNTED_ROI_PCT] = np.divide(
        df[CALC_NPV_RUB],
        cumulative_discounted_costs,
        out=np.full_like(df[CALC_NPV_RUB], 0.0), # Заполняем нулями по умолчанию
        where=cumulative_discounted_costs!=0
    ) * 100
    # Дополнительно обработаем NaN, если они возникли из-за NaN в NPV
    df[CALC_DISCOUNTED_ROI_PCT].fillna(0.0, inplace=True)


    return df

# --- Функция сохранения результатов ---
def save_results(df, output_path):
    """Сохраняет DataFrame с результатами в Excel файл."""
    print(f"Сохранение результатов в {output_path}...")
    try:
        # Упорядочивание столбцов для лучшей читаемости
        input_cols_ordered = [col for col in df.columns if not col.startswith(('Метрика_', 'Эффект_', 'Расчет_')) and col != COL_PERIOD]
        metric_cols_ordered = sorted([col for col in df.columns if col.startswith('Метрика_')])
        effect_cols_ordered = sorted([col for col in df.columns if col.startswith('Эффект_')])
        calc_roi_npv_cols = [
            CALC_TOTAL_BENEFIT_RUB, CALC_ROI_PCT, CALC_PERIOD_YEAR, CALC_DISCOUNT_FACTOR,
            CALC_DISCOUNTED_BENEFIT_RUB, CALC_DISCOUNTED_COST_RUB,
            CALC_DISCOUNTED_CASH_FLOW_RUB, CALC_NPV_RUB, CALC_DISCOUNTED_ROI_PCT
        ]
        calc_roi_npv_cols_existing = [col for col in calc_roi_npv_cols if col in df.columns]

        final_cols_order = ([COL_PERIOD] + input_cols_ordered + metric_cols_ordered +
                            effect_cols_ordered + calc_roi_npv_cols_existing)
        final_cols_order = [col for i, col in enumerate(final_cols_order) if col in df.columns and col not in final_cols_order[:i]]

        df_to_save = df[final_cols_order]
        df_to_save.to_excel(output_path, index=False, engine='openpyxl', float_format="%.2f") # Форматируем числа
        print(f"Результаты успешно сохранены в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов в {output_path}: {e}")
        sys.exit(1)

# --- Функции для построения графиков ---
def plot_deterministic_results(df, plot_prefix):
    """Строит графики NPV и ROI по годам для детерминированного расчета."""
    print(f"Построение графиков детерминированных результатов с префиксом '{plot_prefix}'...")

    # Убедимся, что есть необходимые колонки
    required_cols = [CALC_PERIOD_YEAR, CALC_NPV_RUB, CALC_DISCOUNTED_ROI_PCT, COL_PROJECT_COSTS_RUB]
    if not all(col in df.columns for col in required_cols):
        print("Ошибка: Не найдены все необходимые столбцы для построения графиков.")
        return

    # Игнорируем базовый период (год 0) для некоторых графиков
    df_plot = df[df[CALC_PERIOD_YEAR] > 0].copy()
    if df_plot.empty:
        print("Нет данных для построения графиков (кроме базового периода).")
        return

    # --- График NPV и Инвестиций ---
    fig, ax1 = plt.subplots()

    # Гистограмма инвестиций (номинальных)
    color_bar = 'tab:blue'
    ax1.set_xlabel('Год')
    ax1.set_ylabel('Инвестиции за период (Руб)', color=color_bar)
    bars = ax1.bar(df_plot[CALC_PERIOD_YEAR], df_plot[COL_PROJECT_COSTS_RUB], color=color_bar, alpha=0.6, width=0.4, label='Инвестиции')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' '))) # Формат с пробелами

    # Линия NPV
    ax2 = ax1.twinx() # Вторая ось Y
    color_line = 'tab:red'
    ax2.set_ylabel('NPV накопленный (Руб)', color=color_line)
    line = ax2.plot(df_plot[CALC_PERIOD_YEAR], df_plot[CALC_NPV_RUB], color=color_line, marker='o', label='NPV (накопл.)')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' '))) # Формат с пробелами
    ax2.axhline(0, color='grey', lw=0.5, linestyle='--') # Линия нуля для NPV

    fig.tight_layout() # Чтобы подписи не накладывались
    plt.title('Динамика NPV и Инвестиций по годам')
    # Добавляем общую легенду
    # lns = bars + line # Ошибка, bars - это BarContainer
    lns = [bars] + line
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    plt.savefig(f"{plot_prefix}_npv_investments.png", bbox_inches='tight')
    print(f"График NPV сохранен в {plot_prefix}_npv_investments.png")
    plt.close(fig) # Закрываем фигуру

    # --- График Дисконтированного ROI и Инвестиций ---
    fig, ax1 = plt.subplots()

    # Гистограмма инвестиций (номинальных) - снова для контекста
    color_bar = 'tab:blue'
    ax1.set_xlabel('Год')
    ax1.set_ylabel('Инвестиции за период (Руб)', color=color_bar)
    bars = ax1.bar(df_plot[CALC_PERIOD_YEAR], df_plot[COL_PROJECT_COSTS_RUB], color=color_bar, alpha=0.6, width=0.4, label='Инвестиции')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))

    # Линия Дисконтированного ROI
    ax2 = ax1.twinx()
    color_line = 'tab:green'
    ax2.set_ylabel('Дисконтированный ROI (%)', color=color_line)
    line = ax2.plot(df_plot[CALC_PERIOD_YEAR], df_plot[CALC_DISCOUNTED_ROI_PCT], color=color_line, marker='o', label='Диск. ROI')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.yaxis.set_major_formatter(PercentFormatter()) # Формат процентов
    ax2.axhline(0, color='grey', lw=0.5, linestyle='--') # Линия нуля для ROI

    fig.tight_layout()
    plt.title('Динамика Дисконтированного ROI и Инвестиций по годам')
    # lns = bars + line # Ошибка
    lns = [bars] + line
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    plt.savefig(f"{plot_prefix}_roi_investments.png", bbox_inches='tight')
    print(f"График ROI сохранен в {plot_prefix}_roi_investments.png")
    plt.close(fig)


# --- Функции для графиков симуляции ---
def plot_simulation_distributions(npv_array, roi_array, plot_prefix):
    """Строит гистограммы распределения NPV и ROI по результатам симуляции."""
    print(f"Построение графиков распределения с префиксом '{plot_prefix}'...")

    # График распределения NPV
    plt.figure()
    sns.histplot(npv_array, kde=True, bins=50) # Увеличим кол-во корзин
    plt.title('Распределение итогового NPV (Монте-Карло)')
    plt.xlabel('NPV (Руб)')
    plt.ylabel('Частота')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    mean_npv = np.mean(npv_array)
    median_npv = np.median(npv_array)
    plt.axvline(mean_npv, color='r', linestyle='--', label=f'Среднее: {mean_npv:,.0f}')
    plt.axvline(median_npv, color='g', linestyle=':', label=f'Медиана: {median_npv:,.0f}')
    plt.legend()
    plt.savefig(f"{plot_prefix}_npv_distribution.png", bbox_inches='tight')
    print(f"График распределения NPV сохранен в {plot_prefix}_npv_distribution.png")
    plt.close()

    # График распределения ROI
    plt.figure()
    sns.histplot(roi_array, kde=True, bins=50)
    plt.title('Распределение итогового Дисконтированного ROI (Монте-Карло)')
    plt.xlabel('Дисконтированный ROI (%)')
    plt.ylabel('Частота')
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    mean_roi = np.mean(roi_array)
    median_roi = np.median(roi_array)
    plt.axvline(mean_roi, color='r', linestyle='--', label=f'Среднее: {mean_roi:.1f}%')
    plt.axvline(median_roi, color='g', linestyle=':', label=f'Медиана: {median_roi:.1f}%')
    plt.legend()
    plt.savefig(f"{plot_prefix}_roi_distribution.png", bbox_inches='tight')
    print(f"График распределения ROI сохранен в {plot_prefix}_roi_distribution.png")
    plt.close()

def plot_sensitivity(simulation_df, varied_inputs_map, plot_prefix):
    """Строит график анализа чувствительности (корреляция Спирмена)."""
    print(f"Построение графика анализа чувствительности с префиксом '{plot_prefix}'...")

    if simulation_df.empty:
        print("Нет данных для анализа чувствительности.")
        return

    # Рассчитываем корреляцию Спирмена между входами и выходами (NPV и ROI)
    correlations_npv = {}
    correlations_roi = {}
    input_labels = { # Метки для графика
        'costs': 'Затраты',
        'requests': 'Число запросов',
        'time': 'Время обработки'
    }

    for key, col_name in varied_inputs_map.items():
        # Используем средние значения входов за весь период симуляции
        input_col_name = f'avg_sim_{key}'
        if input_col_name not in simulation_df.columns:
             # Если средние не сохраняли, используем просто ключ (менее точно)
             input_col_name = key

        if input_col_name in simulation_df.columns:
            # Убираем NaN перед расчетом корреляции
            valid_data = simulation_df[[input_col_name, 'final_npv', 'final_roi']].dropna()
            if not valid_data.empty and len(valid_data) > 1: # Корреляция требует > 1 точки
                corr_npv, p_npv = spearmanr(valid_data[input_col_name], valid_data['final_npv'])
                corr_roi, p_roi = spearmanr(valid_data[input_col_name], valid_data['final_roi'])
                label = input_labels.get(key, key) # Используем понятную метку
                # Проверяем на NaN и значимость (p-value < 0.05) - опционально
                correlations_npv[label] = corr_npv if not np.isnan(corr_npv) else 0.0
                correlations_roi[label] = corr_roi if not np.isnan(corr_roi) else 0.0
            else:
                print(f"Предупреждение: Недостаточно данных для расчета корреляции для '{input_col_name}'.")
                correlations_npv[input_labels.get(key, key)] = 0.0
                correlations_roi[input_labels.get(key, key)] = 0.0
        else:
             print(f"Предупреждение: Не найден столбец '{input_col_name}' для анализа чувствительности.")


    if not correlations_npv or not correlations_roi:
        print("Не удалось рассчитать корреляции для анализа чувствительности.")
        return

    # Создаем DataFrame для удобства построения графика
    corr_df_npv = pd.DataFrame(list(correlations_npv.items()), columns=['Параметр', 'Корреляция с NPV']).sort_values('Корреляция с NPV', key=abs, ascending=False)
    corr_df_roi = pd.DataFrame(list(correlations_roi.items()), columns=['Параметр', 'Корреляция с ROI']).sort_values('Корреляция с ROI', key=abs, ascending=False)

    # График для NPV
    plt.figure()
    sns.barplot(x='Корреляция с NPV', y='Параметр', data=corr_df_npv, palette='viridis', hue='Параметр', legend=False)
    plt.title('Анализ чувствительности NPV (Корреляция Спирмена)')
    plt.xlabel('Коэффициент корреляции Спирмена')
    plt.ylabel('')
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_sensitivity_npv.png", bbox_inches='tight')
    print(f"График чувствительности NPV сохранен в {plot_prefix}_sensitivity_npv.png")
    plt.close()

    # График для ROI
    plt.figure()
    sns.barplot(x='Корреляция с ROI', y='Параметр', data=corr_df_roi, palette='viridis', hue='Параметр', legend=False)
    plt.title('Анализ чувствительности Диск. ROI (Корреляция Спирмена)')
    plt.xlabel('Коэффициент корреляции Спирмена')
    plt.ylabel('')
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_sensitivity_roi.png", bbox_inches='tight')
    print(f"График чувствительности ROI сохранен в {plot_prefix}_sensitivity_roi.png")
    plt.close()


# --- Функция для выполнения симуляции Монте-Карло (Векторизованная) ---
def run_monte_carlo_vectorized(df_original, iterations, plot_prefix=None):
    """Выполняет симуляцию Монте-Карло, используя векторизованные операции NumPy."""
    print(f"\nЗапуск векторизованной симуляции Монте-Карло ({iterations} итераций)...")

    num_periods = len(df_original)
    if num_periods == 0:
        print("Ошибка: Входной DataFrame пуст.")
        return

    # Определяем варьируемые входные параметры
    varied_inputs_map = {
        'costs': COL_PROJECT_COSTS_RUB,
        'requests': COL_TOTAL_REQUESTS,
        'time': COL_TOTAL_PROC_TIME_HOURS
    }

    # --- Извлечение базовых данных в NumPy массивы ---
    # Варьируемые параметры (размер: num_periods)
    base_costs = df_original[COL_PROJECT_COSTS_RUB].to_numpy()
    base_requests = df_original[COL_TOTAL_REQUESTS].to_numpy()
    base_proc_time = df_original[COL_TOTAL_PROC_TIME_HOURS].to_numpy()

    # Остальные параметры, необходимые для расчетов (размер: num_periods)
    period_days = df_original[COL_PERIOD_DAYS].to_numpy()
    avg_proc_time_before_ai = df_original[COL_AVG_PROC_TIME_BEFORE_AI_HOURS].to_numpy()
    # total_operating_costs = df_original[COL_TOTAL_OPERATING_COSTS_RUB].to_numpy() # Не используется напрямую в эффектах/ROI
    text_processed_by_ai = df_original[COL_TEXT_PROCESSED_BY_AI].to_numpy() # Не варьируется, но нужно для расчета эффекта
    avg_text_proc_manual = df_original[COL_AVG_TEXT_PROC_MANUAL_HOURS].to_numpy()
    cost_per_hour = df_original[COL_COST_PER_HOUR_RUB].to_numpy()
    discount_rate_pct = df_original[COL_DISCOUNT_RATE_ANNUAL_PCT].to_numpy()

    # --- Генерация случайных вариаций (размер: num_periods x iterations) ---
    cost_std_dev_pct = 0.15
    requests_variation_pct = 0.10
    time_std_dev_pct = 0.10

    # Нормальное распределение для затрат
    sim_costs = np.random.normal(
        loc=base_costs[:, np.newaxis],
        scale=(base_costs * cost_std_dev_pct)[:, np.newaxis],
        size=(num_periods, iterations)
    )
    sim_costs = np.maximum(0, sim_costs) # Затраты >= 0

    # Равномерное распределение для числа запросов
    low_requests = base_requests * (1 - requests_variation_pct)
    high_requests = base_requests * (1 + requests_variation_pct)
    sim_requests = np.random.uniform(
        low=low_requests[:, np.newaxis],
        high=high_requests[:, np.newaxis],
        size=(num_periods, iterations)
    )
    sim_requests = np.maximum(0, sim_requests).astype(int) # Запросы >= 0 и целые

    # Нормальное распределение для времени обработки
    sim_proc_time = np.random.normal(
        loc=base_proc_time[:, np.newaxis],
        scale=(base_proc_time * time_std_dev_pct)[:, np.newaxis],
        size=(num_periods, iterations)
    )
    sim_proc_time = np.maximum(0, sim_proc_time) # Время >= 0

    # --- Векторизованные расчеты для всех итераций ---
    print("Выполнение векторизованных расчетов...")

    # Расширяем неварьируемые параметры до размера (num_periods, iterations)
    avg_proc_time_before_ai_matrix = avg_proc_time_before_ai[:, np.newaxis]
    cost_per_hour_matrix = cost_per_hour[:, np.newaxis]
    avg_text_proc_manual_matrix = avg_text_proc_manual[:, np.newaxis]
    text_processed_by_ai_matrix = text_processed_by_ai[:, np.newaxis]
    discount_rate_matrix = (discount_rate_pct / 100.0)[:, np.newaxis]

    # Метрики (только необходимые для эффектов)
    metric_avg_proc_time = safe_divide_np(sim_proc_time, sim_requests)
    metric_hours_saved = (avg_proc_time_before_ai_matrix - metric_avg_proc_time) * sim_requests
    metric_hours_saved = np.maximum(0, metric_hours_saved) # clip(lower=0)

    # Эффекты
    effect_fot_savings = metric_hours_saved * cost_per_hour_matrix
    effect_text_proc_savings = text_processed_by_ai_matrix * avg_text_proc_manual_matrix * cost_per_hour_matrix

    # Совокупная выгода
    total_benefit = effect_fot_savings + effect_text_proc_savings

    # NPV и Дисконтированный ROI
    period_year = (np.arange(num_periods) * 0.5)[:, np.newaxis] # (num_periods, 1)
    discount_factor = 1 / (1 + discount_rate_matrix)**period_year

    discounted_benefit = total_benefit * discount_factor
    discounted_cost = sim_costs * discount_factor
    discounted_cash_flow = discounted_benefit - discounted_cost

    # NPV (накопленный) - кумулятивная сумма по оси периодов (axis=0)
    npv = np.cumsum(discounted_cash_flow, axis=0)

    # Дисконтированный ROI
    cumulative_discounted_costs = np.cumsum(discounted_cost, axis=0)
    discounted_roi = safe_divide_np(npv, cumulative_discounted_costs) * 100

    # --- Извлечение итоговых результатов симуляции ---
    final_npvs = npv[-1, :] # Последний период для всех итераций
    final_discounted_rois = discounted_roi[-1, :] # Последний период для всех итераций

    # --- Сбор данных для анализа чувствительности ---
    simulation_results_list = []
    # Сохраняем средние значения симулированных входов по периодам для каждой итерации
    avg_sim_costs = np.mean(sim_costs, axis=0)
    avg_sim_requests = np.mean(sim_requests, axis=0)
    avg_sim_time = np.mean(sim_proc_time, axis=0)

    for i in tqdm(range(iterations), desc="Сбор данных для анализа"): # Добавили tqdm сюда
        sim_data_point = {
            'iteration': i,
            'final_npv': final_npvs[i],
            'final_roi': final_discounted_rois[i],
            'avg_sim_costs': avg_sim_costs[i],
            'avg_sim_requests': avg_sim_requests[i],
            'avg_sim_time': avg_sim_time[i]
        }
        simulation_results_list.append(sim_data_point)
    simulation_results_df = pd.DataFrame(simulation_results_list)


    # --- Расчет и вывод статистик ---
    print("\n--- Результаты симуляции Монте-Карло ---")
    if final_npvs.size > 0 and final_discounted_rois.size > 0:
        print("\nСтатистики для Итогового NPV:")
        print(f"  Среднее: {np.mean(final_npvs):,.2f} Руб")
        print(f"  Медиана: {np.median(final_npvs):,.2f} Руб")
        print(f"  Стандартное отклонение: {np.std(final_npvs):,.2f} Руб")
        print(f"  5-й перцентиль: {np.percentile(final_npvs, 5):,.2f} Руб")
        print(f"  95-й перцентиль: {np.percentile(final_npvs, 95):,.2f} Руб")
        print(f"  Вероятность положительного NPV: {np.mean(final_npvs > 0) * 100:.2f}%")

        print("\nСтатистики для Итогового Дисконтированного ROI:")
        print(f"  Среднее: {np.mean(final_discounted_rois):.2f}%")
        print(f"  Медиана: {np.median(final_discounted_rois):.2f}%")
        print(f"  Стандартное отклонение: {np.std(final_discounted_rois):.2f}%")
        print(f"  5-й перцентиль: {np.percentile(final_discounted_rois, 5):.2f}%")
        print(f"  95-й перцентиль: {np.percentile(final_discounted_rois, 95):.2f}%")

        # Построение графиков симуляции, если указан префикс
        if plot_prefix:
            plot_simulation_distributions(final_npvs, final_discounted_rois, plot_prefix)
            # Передаем DataFrame с результатами симуляции для анализа чувствительности
            plot_sensitivity(simulation_results_df, varied_inputs_map, plot_prefix)
    else:
        print("Не удалось собрать результаты симуляций.")


def main():
    """Главная функция скрипта."""
    parser = argparse.ArgumentParser(description="Расчет эффектов и ROI внедрения ИИ по данным из CSV, с опциональной симуляцией Монте-Карло.")
    parser.add_argument("input_csv", help="Путь к входному CSV файлу с данными.")
    parser.add_argument("-o", "--output", default="results.xlsx",
                        help="Путь к выходному Excel файлу для сохранения результатов детерминированного расчета (по умолчанию: results.xlsx).")
    parser.add_argument("--simulate", action="store_true",
                        help="Включить режим симуляции Монте-Карло.")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Количество итераций для симуляции Монте-Карло (по умолчанию: 10000).")
    parser.add_argument("--plot-prefix", type=str, default=None,
                        help="Префикс для имен файлов с графиками (например, 'my_project'). Если указан, графики будут сохранены.")

    args = parser.parse_args()

    # 1. Загрузка данных
    data_df = load_data(args.input_csv)

    if args.simulate:
        # Выполняем векторизованную симуляцию Монте-Карло
        run_monte_carlo_vectorized(data_df, args.iterations, args.plot_prefix)
    else:
        # Выполняем стандартный детерминированный расчет
        print("Выполнение детерминированного расчета...")
        # Используем _df версии функций
        metrics_df = calculate_metrics_df(data_df.copy())
        effects_df = calculate_economic_effects_df(metrics_df)
        roi_npv_df = calculate_roi_npv_df(effects_df)

        save_results(roi_npv_df, args.output)
        print("Детерминированный расчет завершен.")

        # Строим графики для детерминированного расчета, если указан префикс
        if args.plot_prefix:
            plot_deterministic_results(roi_npv_df, args.plot_prefix)

    print("\nРабота скрипта завершена.")

if __name__ == "__main__":
    main()
