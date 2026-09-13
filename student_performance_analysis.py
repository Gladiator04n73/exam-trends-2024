# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import export_graphviz
import graphviz
from imblearn.over_sampling import SMOTE


df = pd.read_csv('StudentPerformanceFactors.csv')

df

"""Этот набор данных содержит информацию об академической успеваемости, образе жизни и социально-экономических факторах, связанных со студентами, в том числе об их результатах на экзаменах.

Он включает в себя данные о времени, потраченном на учебу, посещаемости, продолжительности сна, мотивации, дополнительных занятиях, участии родителей, типе учебного заведения и других факторах окружающей среды.

Целевой переменной является Exam_Score, которая отражает итоговую оценку студента на экзамене

'Hours_Studied' - часы обучения (количественная)

'Attendance' - Посещаемость (количественная)

'Parental_Involvement' - «Участие родителей»  (категориальная Medium, High, Low)

'Access_to_Resources' - «Доступ к ресурсам» (категориальная Medium, High, Low)

'Extracurricular_Activities' - «Внеклассные мероприятия» (категориальная YES, NO)

'Sleep_Hours', - «Часы сна» (количественная)

'Previous_Scores' - «Предыдущие оценки» (количественная)
       
'Motivation_Level' - «Уровень мотивации» (категориальная Medium, High, Low)

'Internet_Access', - «Доступ к интернету» (категориальная YES, NO)

'Tutoring_Sessions' - «Репетиторские занятия» (количественная)

'Family_Income', - «Доход семьи» (категориальная Medium, High, Low)

'Teacher_Quality' - «Квалификация учителя» (категориальная Medium, High, Low)

'School_Type', - «Тип школы» (категориальная Public, Private)

'Peer_Influence' - «Влияние сверстников» (категориальная Positive, Neutral,	Negative)

'Physical_Activity', - «Уровень активность» (количественная)

'Learning_Disabilities', - «Нарушения обучаемости» (категориальная YES, NO)

'Parental_Education_Level' - «Уровень образования родителей» (категориальная High School, College, Postgraduate)

'Distance_from_Home', - «Расстояние от дома» (категориальная Near, Moderate, Far - Ближний, Средний, Дальний)

'Gender', - Пол (категориальная Male, Female)

'Exam_Score' - «Результаты экзаменов» (количественная)

Описательная статистика
"""

df.describe()

"""Студенты в среднем спят по 7 часов

Среднее значение экзаменов 67 баллов, максимальный балл 101, минимальный 55

В данной выборке 235 пропусков (78 пропусков по параметру Teacher_Quality, 90 пропусков по параметру Parental_Education_Level, 67 пропусков по параметру Distance_from_Home) => у 4 студентов учителя и родители без образования
"""

df.describe(include='object')

"""Мужчин - 3814, женщин - 2793

большинство студентов участвую во внеклассных мероприятиях 3938, у 5912 учащихся нет нарушения в обучаемости, 3884 учащихся живут недалеко от школы, 4598 студентов учатся в средней школе, у 2672 учащихся родители имеют низкий доход
"""

display(df['Parental_Involvement'].value_counts(),df['Teacher_Quality'].value_counts(),
df['Gender'].value_counts())

quantitative_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

print("--- Группировка количественных признаков по категориальным ---")
for cat_col in categorical_cols:
    print(f"\nГруппировка по: {cat_col}")
    grouped_data = df.groupby(cat_col)[quantitative_cols].mean()
    display(grouped_data)
    print("="*50)

"""Для анализа связи между категориальными и количественными переменными, давайте сгруппируем данные по каждой категориальной колонке и вычислим средние значения для всех количественных колонок."""

for cat_col in categorical_cols:
    print(f"\n--- Группировка по: {cat_col} ---")
    grouped_data = df.groupby(cat_col)[quantitative_cols].mean()
    display(grouped_data)
    print("\n" + "="*50 + "\n")

"""Разделим столбцы на категориальные и количественные переменные:"""

plt.figure(figsize=(6,5))
sns.countplot(data=df, x = "Parental_Involvement", palette = "husl")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Участие родителей (низкое, среднее, высокое)")
plt.ylabel("Количество")
plt.title("Распределение участия родителей в образовательном процессе")
plt.savefig(f'images/plot_01.png', dpi=150, bbox_inches='tight')
plt.show()

df[df['Tutoring_Sessions']==0 & df['Teacher_Quality'].isna()]

df['Tutoring_Sessions'][df['Teacher_Quality'].isna()]

df.isna().sum()

df[df['Teacher_Quality'].isna() & df['Parental_Education_Level'].isna()]

corr = df.corr(numeric_only=True)
plt.figure(figsize=(14, 12))
with sns.axes_style("white"):
  ax = sns.heatmap(corr, annot=True, linewidths=.2, fmt=".2f")
plt.savefig(f'images/plot_02.png', dpi=150, bbox_inches='tight')
plt.show()

"""Имеется положительная корреляционная зависимость между Результатами экзаменов и Посещаемостью и Результатами экзаменов и Часами обучения"""

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(y=df[df['Gender'] == 'Male']['Exam_Score'], ax=axes[0])
axes[0].set_title('Распределение результатов экзаменов для мужчин')
axes[0].set_ylabel('Баллы по экзамену')
axes[0].set_xlabel('Мужчины')

sns.boxplot(y=df[df['Gender'] == 'Female']['Exam_Score'], ax=axes[1])
axes[1].set_title('Распределение результатов экзаменов для женщин')
axes[1].set_ylabel('Баллы по экзамену')
axes[1].set_xlabel('Женщины')

plt.tight_layout()
plt.savefig(f'images/plot_03.png', dpi=150, bbox_inches='tight')
plt.show()

plt.scatter(df['Hours_Studied'], df['Exam_Score'])
plt.xlabel('Часы обучения')
plt.ylabel('Баллы по экзамену')
plt.savefig(f'images/plot_04.png', dpi=150, bbox_inches='tight')
plt.show()

plt.scatter(df['Attendance'], df['Exam_Score'])
plt.xlabel('Посещаемость')
plt.ylabel('Баллы по экзамену')
plt.savefig(f'images/plot_05.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
sns.stripplot(x='Gender', y='Exam_Score', data=df, jitter=True, hue='Gender', palette='viridis', legend=False)
plt.title('Распределение баллов по экзаменам')
plt.xlabel('Пол')
plt.ylabel('Баллы за экзамен')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'images/plot_06.png', dpi=150, bbox_inches='tight')
plt.show()

"""Мужчины и женщины имеют одинаковые средние баллы за экзамен. Аномально высокие баллы наблюдаются у женщин"""

sns.boxplot(y = df['Attendance']).set_ylabel("Посещения")

Exam_Score_vs_Teacher_Quality=sns.boxplot(y = df['Exam_Score'], x = df['Teacher_Quality'])
Exam_Score_vs_Teacher_Quality.set_ylabel("Баллы за экзамен")
Exam_Score_vs_Teacher_Quality.set_xlabel("Квалификация учителя")

df_0=df[df['Gender']=='Male']
df_1=df[df['Gender']=='Female']

df_0.hist('Exam_Score', bins=20),df_0.shape

df_1.hist('Exam_Score', bins=20),df_1.shape

"""Среднее значения результатов экзаменов между мужчинами и женщинами не отличается, но у женщин больше представителей с баллами свыше 90"""

df.describe()

df_num = df[['Hours_Studied', 'Attendance', 'Previous_Scores',
        'Sleep_Hours','Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']]

print("df_num before normalization (first 5 rows):")
display(df_num.head())
print("\ndf_num before normalization (last 5 rows):")
display(df_num.tail())

scaler = MinMaxScaler()

X = df_num[['Attendance', 'Hours_Studied', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions',
            'Physical_Activity', 'Exam_Score']]

X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

print("Min-Max Scaled features for clustering (X_norm - first 5 rows):")
display(X_norm.head())

print("\nMin-Max Scaled features for clustering (X_norm - last 5 rows):")
display(X_norm.tail())

d = linkage(X_norm, method='complete')

# Строим дендрограмму
plt.figure(figsize=(9, 6))
dendrogram(d, orientation='top')
plt.title('Дендрограмма для иерархической кластеризации (метод complete)')
plt.xlabel('Индексы объектов или размер кластера')
plt.ylabel('Расстояние (Height)')
plt.savefig(f'images/plot_07.png', dpi=150, bbox_inches='tight')
plt.show()

d = linkage(X_norm, method='weighted')
plt.figure(figsize=(9, 6))
dendrogram(d, orientation='top')
plt.title('Дендрограмма для иерархической кластеризации (метод weighted)')
plt.xlabel('Индексы объектов или размер кластера')
plt.ylabel('Расстояние (Height)')
plt.savefig(f'images/plot_08.png', dpi=150, bbox_inches='tight')
plt.show()

d = linkage(X_norm, method='ward')
plt.figure(figsize=(9, 6))
dendrogram(d, orientation='top')
plt.title('Дендрограмма для иерархической кластеризации (метод ward)')
plt.xlabel('Индексы объектов или размер кластера')
plt.ylabel('Расстояние (Height)')
plt.savefig(f'images/plot_09.png', dpi=150, bbox_inches='tight')
plt.show()

"""Дендограммы построеные по принципами средней связи, варда, cредневзвешенному расстоянию дали четкое разбиение на 3 кластера"""

# создадим пустой список для записи функционала f
f = []
# воспользуемся функцией range(), возвращающей последовательность чисел от 1 до 10
for i in range(1, 11):
    # настроим параметры модели
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 42)
    # обучим модель на нормированных данных для разного числа кластеров
    kmeans.fit(X_norm)
     # для каждого кластера рассчитаем функционал (атрибут inertia_) и поместим в список
    f.append(kmeans.inertia_)
# зададим размер график
plt.figure(figsize = (10,6))
# передадим функции plot() последовательность кластеров и функционал
plt.plot(range(1, 11), f)
# задаем названия графика и осей
plt.title('Выбор оптимального количества кластеров методом локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Функционал')

"""Метод логтя определил 4 кластера"""

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 42)
y_pred = kmeans.fit_predict(X_norm)
clusters_1 = kmeans.labels_
df_num['number_KM'] = clusters_1
display(df_num.head())

plt.figure(figsize=(10, 7))
custom_palette = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

# Re-create a temporary df for K-Means plotting to ensure 'number_KM' column exists
df_for_km_plot = df_num[['Attendance', 'Exam_Score']].copy()
df_for_km_plot['number_KM'] = clusters_1

sns.scatterplot(x=X_norm['Attendance'], y=X_norm['Exam_Score'],
                hue=df_for_km_plot['number_KM'], palette=custom_palette, s=100, alpha=0.7)
plt.title('K-Means')
plt.xlabel('Посещаемость')
plt.ylabel('Баллы за экзамен')

# Get cluster counts
cluster_counts = df_for_km_plot['number_KM'].value_counts().sort_index()

# Create custom legend labels with counts
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = []
for label_val in sorted(df_for_km_plot['number_KM'].unique()):
    count = cluster_counts.get(label_val, 0)
    new_labels.append(f'{label_val} (n={count})')

# Adjust handles to match the number of clusters, skipping the 'hue' label if present
if len(labels) > len(new_labels):
    plt.legend(handles[1:], new_labels, title='Кластер', loc='upper left')
else:
    plt.legend(handles, new_labels, title='Кластер', loc='upper left')

plt.grid(True)
plt.savefig(f'images/plot_10.png', dpi=150, bbox_inches='tight')
plt.show()

df_num['number_KM'].value_counts()

plt.figure(figsize=(10, 7))
custom_palette = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

# Create a temporary df for K-Means plotting to ensure 'number_KM' column exists and is aligned
df_for_km_plot = df_num[['Hours_Studied', 'Exam_Score']].copy()
df_for_km_plot['number_KM'] = clusters_1

sns.scatterplot(x=X_norm['Hours_Studied'], y=X_norm['Exam_Score'],
                hue=df_for_km_plot['number_KM'], palette=custom_palette,
                s=100, alpha=0.7)
plt.title('K-Means')
plt.xlabel('Часы обучения')
plt.ylabel('Баллы за экзамен')

# Get cluster counts
cluster_counts = df_for_km_plot['number_KM'].value_counts().sort_index()

# Create custom legend labels with counts
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = []
for label_val in sorted(df_for_km_plot['number_KM'].unique()):
    count = cluster_counts.get(label_val, 0)
    new_labels.append(f'{label_val} (n={count})')

# Adjust handles to match the number of clusters, skipping the 'hue' label if present
if len(labels) > len(new_labels):
    plt.legend(handles[1:], new_labels, title='Кластер', loc='upper left')
else:
    plt.legend(handles, new_labels, title='Кластер', loc='upper left')

plt.grid(True)
plt.savefig(f'images/plot_11.png', dpi=150, bbox_inches='tight')
plt.show()

"""Путем кластеризаци были получены 4 кластера.

Ученики с высокой посещаемостью кластеры 0 и 3

Ученики с низкой посещаемостью кластеры 1 и 2

Участники 0 и 2 кластера имеют больше учеников с высокими баллами

Ученики из 3 кластера самые неактивные (с маленькими посещаниями и часами обучения)
"""

df_num

"""На рисунке видно, что метод DBScam всех учеников определил в 1 кластер, т.к. большинство учеников имеют средние баллы за экзамен. Метод определил учеников с высокими баллами, как шумовых"""

AggCl = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
y_pred = AggCl.fit_predict(X_norm)
clusters_3 = AggCl.labels_
clusters_3

# Re-define df_num with relevant columns for clustering
df_num = df[['Hours_Studied', 'Attendance', 'Previous_Scores',
        'Sleep_Hours','Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']].copy()

# Re-normalize X_norm
scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns, index=df_num.index)

# Re-run Agglomerative Clustering
AggCl = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
clusters_3 = AggCl.fit_predict(X_norm)

# Create a temporary df for Agglomerative Clustering plotting to ensure 'number_AC' column exists and is aligned
df_for_ac_plot = df_num[['Hours_Studied', 'Exam_Score']].copy()
df_for_ac_plot['number_AC'] = clusters_3

plt.figure(figsize=(10, 7))

sns.scatterplot(x=X_norm['Hours_Studied'], y=X_norm['Exam_Score'],
                hue=df_for_ac_plot['number_AC'], # Use the temporary DataFrame here
                palette='viridis', # Using a different palette for visual distinction
                s=100, alpha=0.7)
plt.title('Агломеративная Кластеризация')
plt.xlabel('Часы обучения')
plt.ylabel('Баллы за экзамен')

# Get cluster counts
cluster_counts = df_for_ac_plot['number_AC'].value_counts().sort_index()

# Create custom legend labels with counts
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = []
for label_val in sorted(df_for_ac_plot['number_AC'].unique()):
    count = cluster_counts.get(label_val, 0)
    new_labels.append(f'{label_val} (n={count})')

# Adjust handles to match the number of clusters, skipping the 'hue' label if present
if len(labels) > len(new_labels):
    plt.legend(handles[1:], new_labels, title='Кластер', loc='upper left')
else:
    plt.legend(handles, new_labels, title='Кластер', loc='upper left')

plt.grid(True)
plt.savefig(f'images/plot_12.png', dpi=150, bbox_inches='tight')
plt.show()

df_num['number_AC'] = clusters_3
df_num

ac_class_0 = df_num.loc[df_num['number_AC'] == 0]
ac_class_1 = df_num.loc[df_num['number_AC'] == 1]
ac_class_2 = df_num.loc[df_num['number_AC'] == 2]

ac_type_0=ac_class_0.groupby(['number_AC']).agg({'number_AC' : ['count']})
ac_type_1=ac_class_1.groupby(['number_AC']).agg({'number_AC' : ['count']})
ac_type_2=ac_class_2.groupby(['number_AC']).agg({'number_AC' : ['count']})

display(ac_class_0,ac_class_1,ac_class_2)

silhouette_1 = silhouette_score(X_norm, clusters_1)
dbscan = DBSCAN(eps=0.5, min_samples=5) # Default parameters, may need tuning
clusters_2 = dbscan.fit_predict(X_norm)
silhouette_2 = silhouette_score(X_norm, clusters_2)
silhouette_3 = silhouette_score(X_norm, clusters_3)
print('Метрика силуэта (метод к-средних):', silhouette_1.round(3))
print('Метрика силуэта (метод DBSCAN):', silhouette_2.round(3))
print('Метрика силуэта (метод агломеративной кластеризации):', silhouette_3.round(3))

CHS_1 = calinski_harabasz_score(X_norm, clusters_1)
CHS_2 = calinski_harabasz_score(X_norm, clusters_2)
CHS_3 = calinski_harabasz_score(X_norm, clusters_3)
print('Метрика силуэта (метод к-средних):', CHS_1.round(3))
print('Метрика силуэта (метод DBSCAN):', CHS_2.round(3))
print('Метрика силуэта (метод агломеративной кластеризации):', CHS_3.round(3))

DBS_1 = davies_bouldin_score(X_norm, clusters_1)
DBS_2 = davies_bouldin_score(X_norm, clusters_2)
DBS_3 = davies_bouldin_score(X_norm, clusters_3)
print('Метрика силуэта (метод к-средних):', DBS_1.round(3))
print('Метрика силуэта (метод DBSCAN):', DBS_2.round(3))
print('Метрика силуэта (метод агломеративной кластеризации):', DBS_3.round(3))

ac_clusters_means = df_num.groupby('number_AC')[['Attendance', 'Hours_Studied',
                                                'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions',
                                                'Physical_Activity', 'Exam_Score']].mean()
display(ac_clusters_means)

df_num['number_DB'] = clusters_2
db_clusters_means = df_num.groupby('number_DB')[['Attendance', 'Hours_Studied',
                                                'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions',
                                                'Physical_Activity', 'Exam_Score']].mean()
display(db_clusters_means)

plt.figure(figsize=(10, 7))

sns.scatterplot(x=X_norm['Hours_Studied'], y=X_norm['Exam_Score'],
                hue=df_num['number_DB'],
                s=100, alpha=0.7)
plt.title('DBSCAN Кластеризация')
plt.xlabel('Часы обучения')
plt.ylabel('Баллы за экзамен')
plt.legend(title='Кластер DBSCAN')
plt.grid(True)
plt.savefig(f'images/plot_13.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 7))

sns.scatterplot(x=X_norm['Hours_Studied'], y=X_norm['Exam_Score'],
                hue=df_num['number_DB'],
                s=100, alpha=0.7)
plt.title('DBSCAN Кластеризация')
plt.xlabel('Часы обучения')
plt.ylabel('Баллы за экзамен')
plt.legend(title='Кластер DBSCAN')
plt.grid(True)
plt.savefig(f'images/plot_14.png', dpi=150, bbox_inches='tight')
plt.show()

# Ensure df_num is correctly defined for clustering and has the 'number_KM' column
df_num = df[['Hours_Studied', 'Attendance', 'Previous_Scores',
        'Sleep_Hours','Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']].copy()

# Re-run K-Means to ensure clusters_1 is available and consistent
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 42)
clusters_1 = kmeans.fit_predict(X_norm) # Assuming X_norm is defined from previous steps

df_num['number_KM'] = clusters_1

km_clusters_means = df_num.groupby('number_KM')[['Attendance', 'Hours_Studied',
                                                'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions',
                                                'Physical_Activity', 'Exam_Score']].mean()
display(km_clusters_means)

metrics_data = {
    'Метод': ['K-Means', 'DBSCAN', 'Агломеративная Кластеризация'],
    'Silhouette Score': [silhouette_1.round(2), silhouette_2.round(2), silhouette_3.round(2)],
    'Calinski-Harabasz Score': [CHS_1.round(2), CHS_2.round(2), CHS_3.round(2)],
    'Davies-Bouldin Score': [DBS_1.round(2), DBS_2.round(2), DBS_3.round(2)]
}

metrics_df = pd.DataFrame(metrics_data)

display(metrics_df)

display(km_clusters_means)

"""###Классификация

###3 класса
"""

conditions = [
    df['Exam_Score'] <= 65,
    (df['Exam_Score'] > 65) & (df['Exam_Score'] <= 75),
    df['Exam_Score'] > 75
]
choices = [0, 1, 2]
df['knowledge_assessment'] = np.select(conditions, choices, default=np.nan)
df

df.groupby('knowledge_assessment')['knowledge_assessment'].count()

df_num = df[['Hours_Studied', 'Attendance', 'Sleep_Hours',
       'Previous_Scores',
       'Tutoring_Sessions',
       'Physical_Activity',
       'Exam_Score', 'knowledge_assessment']]

y = df_num['knowledge_assessment']
X = df_num.drop(['knowledge_assessment', 'Exam_Score'], axis=1)

"""###Логистрическая регрессия"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
model = LogisticRegression(solver='liblinear', penalty='l2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matr = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_101.png', dpi=150, bbox_inches='tight')

"""Модель показывает хорошее качество на классах 0 и 1. Доля верных предсказаний составляет 76% для класса 0 и 92% для класса 1.

Однако класс 2 модель не распознаёт ни разу — все 18 объектов этого класса ошибочно отнесены к классам 0 или 1. Причина в сильном дисбалансе выборки.

Класс 2 составляет менее 2% от общего числа объектов, из-за чего модель не находит устойчивых закономерностей для его выделения и минимизирует общую ошибку, полностью игнорируя редкий класс. Это говорит о необходимости балансировки классов (например, через oversampling) перед дальнейшим обучением.
"""

print(classification_report(y_test, y_pred, target_names=['0','1','2']))

"""Общая accuracy (0.86) выглядит высокой, но она вводит в заблуждение — она рассчитана по всей выборке и почти полностью определяется классами 0 и 1, которые вместе составляют 98.6% данных (1304 из 1322). Более честную картину даёт macro avg (0.56 по всем метрикам) — среднее по классам без учёта их размера, которое явно показывает провал по классу 2.

###Деревья решений
"""

tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_model.fit(X_train,y_train)
y_pred_1 = tree_model.predict(X_test)
print(classification_report(y_test,y_pred_1))

export_graphviz(tree_model.fit(X_train, y_train), out_file="tree.dot")
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

tree_model_1 = DecisionTreeClassifier(max_depth=10, random_state=0)
tree_model_1.fit(X_train,y_train)
y_pred_1 = tree_model_1.predict(X_test)
print(classification_report(y_test,y_pred_1))
conf_matr = confusion_matrix(y_test,y_pred_1)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_102.png', dpi=150, bbox_inches='tight')

"""Дерево решений выделяет посещаемость как главный признак для разделения классов. Вторым по значимости оказывается количество часов обучения.

Наиболее уверенно модель разделяет крайние случаи: студенты с низкой посещаемостью и малым количеством часов учёбы почти гарантированно попадают в класс с низким баллом, а студенты с высокой посещаемостью, большим количеством часов учёбы и хорошими предыдущими оценками — в класс с высоким баллом. При этом уже в корне дерева виден дисбаланс классов (865 против 4420), что ограничивает способность модели уверенно предсказывать редкий класс в промежуточных, менее однозначных случаях.

###Случайный лес
"""

RF_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,          # ограничить глубину деревьев
    min_samples_split=20, # минимум объектов для разбиения узла
    min_samples_leaf=10,  # минимум объектов в листе
    max_features='sqrt',  # меньше признаков на каждое дерево
    random_state=0
)
RF_model.fit(X_train,y_train)
y_pred_2 = RF_model.predict(X_test)
print(classification_report(y_test,y_pred_2))
conf_matr = confusion_matrix(y_test,y_pred_2)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_103.png', dpi=150, bbox_inches='tight')

"""Модель переобучилась, метод не подходит для данной выборки.

###Градиентный бустинг
"""

GB_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,           # у бустинга деревья должны быть неглубокими
    learning_rate=0.05,    # маленький шаг обучения
    subsample=0.8,         # обучение на подвыборке снижает переобучение
    random_state=0
)
GB_model.fit(X_train,y_train)
y_pred_3 = GB_model.predict(X_test)
print(classification_report(y_test,y_pred_3))
conf_matr = confusion_matrix(y_test,y_pred_3)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_104.png', dpi=150, bbox_inches='tight')

"""Модель переобучилась, метод не подходит для данной выборки.

###Балансировка
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
ros = RandomOverSampler(random_state=0)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

model_up = LogisticRegression(solver='liblinear', penalty='l2')
model_up.fit(X_train_res, y_train_res)
y_up_pred = model_up.predict(X_test)
print(classification_report(y_test,y_up_pred))
conf_matr = confusion_matrix(y_test,y_up_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_105.png', dpi=150, bbox_inches='tight')

"""| Метод                   | Precision (редкий класс) | Recall (редкий класс) | Accuracy | Macro F1 |

|-------------------------|--------------------------|-----------------------|----------|----------|

| Без балансировки        | 0.00                     | 0.00                  | 0.86     | 0.56     |

| RandomOverSampler       | 0.02                     | 0.47                  | 0.59     | 0.48     |

| SMOTE                   | 0.01                     | 0.33                  | 0.61     | 0.49     |

| Частичный oversampling  | 0.00                     | 0.00                  | 0.87     | 0.57     |

| class_weight='balanced' | 0.00                     | 0.00                  | 0.87     | 0.57     |
"""

smote = SMOTE(random_state=0, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model_up = LogisticRegression(solver='liblinear', penalty='l2')
model_up.fit(X_train_res, y_train_res)
y_up_pred = model_up.predict(X_test)
print(classification_report(y_test,y_up_pred))
conf_matr = confusion_matrix(y_test,y_up_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_106.png', dpi=150, bbox_inches='tight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model_balanced = LogisticRegression(solver='liblinear', penalty='l2', class_weight='balanced')
model_balanced.fit(X_train, y_train)
y_balanced_pred = model_balanced.predict(X_test)
print(classification_report(y_test, y_balanced_pred))

conf_matr = confusion_matrix(y_test, y_balanced_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_107.png', dpi=150, bbox_inches='tight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

ros_partial = RandomOverSampler(sampling_strategy={2: 100}, random_state=0)
X_train_res, y_train_res = ros_partial.fit_resample(X_train, y_train)

model_partial = LogisticRegression(solver='liblinear', penalty='l2')
model_partial.fit(X_train_res, y_train_res)
y_partial_pred = model_partial.predict(X_test)
print(classification_report(y_test, y_partial_pred))

conf_matr = confusion_matrix(y_test, y_partial_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_108.png', dpi=150, bbox_inches='tight')

"""Метод балансировки SMOTE не дал улучшений для модели, при столь малом количестве объектов редкого класса 2

###2 класса
"""

conditions = [
    df['Exam_Score'] <= 70,
    (df['Exam_Score'] > 70)
]
choices = [0, 1]
df['knowledge_assessment'] = np.select(conditions, choices, default=np.nan)
df.groupby('knowledge_assessment')['knowledge_assessment'].count()

df_num = df[['Hours_Studied', 'Attendance', 'Sleep_Hours',
       'Previous_Scores',
       'Tutoring_Sessions',
       'Physical_Activity',
       'Exam_Score', 'knowledge_assessment']]

y = df_num['knowledge_assessment']
X = df_num.drop(['knowledge_assessment', 'Exam_Score'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
model = LogisticRegression(solver='liblinear', penalty='l2')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matr = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_109.png', dpi=150, bbox_inches='tight')

tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_model.fit(X_train,y_train)
y_pred_1 = tree_model.predict(X_test)
print(classification_report(y_test,y_pred_1))
conf_matr = confusion_matrix(y_test,y_pred_1)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_110.png', dpi=150, bbox_inches='tight')

export_graphviz(tree_model.fit(X_train, y_train), out_file="tree.dot")
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

RF_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=0
)
RF_model.fit(X_train,y_train)
y_pred_2 = RF_model.predict(X_test)
print(classification_report(y_test,y_pred_2))
conf_matr = confusion_matrix(y_test,y_pred_2)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_111.png', dpi=150, bbox_inches='tight')

GB_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=0
)
GB_model.fit(X_train,y_train)
y_pred_3 = GB_model.predict(X_test)
print(classification_report(y_test,y_pred_3))
conf_matr = confusion_matrix(y_test,y_pred_3)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_112.png', dpi=150, bbox_inches='tight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
ros = RandomOverSampler(random_state=0)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
model_up = LogisticRegression(solver='liblinear', penalty='l2')
model_up.fit(X_train_res, y_train_res)
y_up_pred = model_up.predict(X_test)
print(classification_report(y_test,y_up_pred))
conf_matr = confusion_matrix(y_test,y_up_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_113.png', dpi=150, bbox_inches='tight')

smote = SMOTE(random_state=0, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
model_up = LogisticRegression(solver='liblinear', penalty='l2')
model_up.fit(X_train_res, y_train_res)
y_up_pred = model_up.predict(X_test)
print(classification_report(y_test,y_up_pred))
conf_matr = confusion_matrix(y_test,y_up_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_114.png', dpi=150, bbox_inches='tight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

ros_partial = RandomOverSampler(sampling_strategy={1: 2000}, random_state=0)
X_train_res, y_train_res = ros_partial.fit_resample(X_train, y_train)

model_partial = LogisticRegression(solver='liblinear', penalty='l2')
model_partial.fit(X_train_res, y_train_res)
y_partial_pred = model_partial.predict(X_test)
print(classification_report(y_test, y_partial_pred))

conf_matr = confusion_matrix(y_test, y_partial_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_115.png', dpi=150, bbox_inches='tight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model_balanced = LogisticRegression(solver='liblinear', penalty='l2', class_weight='balanced')
model_balanced.fit(X_train, y_train)
y_balanced_pred = model_balanced.predict(X_test)
print(classification_report(y_test, y_balanced_pred))

conf_matr = confusion_matrix(y_test, y_balanced_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанное значение')
plt.ylabel('Фактическое значение')
plt.title('Матрица ошибок')
plt.savefig(f'images/confusion_matrix_116.png', dpi=150, bbox_inches='tight')

"""Сравнение задач бинарной (2 класса) и многоклассовой (3 класса) классификации показало, что модель на 2 классах даёт значительно более высокое и стабильное качество.

Причина разрыва — не в самом количестве классов, а в распределении объектов, которое возникает при выбранных порогах разбиения Exam_Score. При делении на 2 класса редкий класс получил 866 объектов в обучающей выборке, что оказалось достаточно для устойчивого обучения логистической регрессии. При делении на 3 класса один из классов содержал всего около 18 объектов — этого объёма недостаточно для обучения ни при одном из протестированных методов балансировки
"""

methods = ['RandomOverSampler', 'SMOTE']
recall_class2 = [0.47, 0.22]
precision_class2 = [0.02, 0.01]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, recall_class2, width, label='Recall (класс 2)', color='seagreen')
bars2 = ax.bar(x + width/2, precision_class2, width, label='Precision (класс 2)', color='indianred')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение методов балансировки по редкому классу')
ax.legend()

plt.tight_layout()
plt.savefig(f'images/plot_15.png', dpi=150, bbox_inches='tight')
plt.show()