# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

##################################################
# import modules

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import seaborn as sns
from scipy import stats

# import requests
# from tqdm import tqdm
# from shapely.geometry import Polygon
# from pyproj import CRS
# from matplotlib.colors import LinearSegmentedColormap
# import seaborn as sns
# from collections import Counter
# import re

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# Set the max_columns option to None
pd.set_option('display.max_columns', None)

##################################################
# set working directory

cwd = os.getcwd()
print(cwd)

base_path = "/Users/jaehojung/PycharmProjects/genderSortingAcrossElementarySchoolInKorea"

path_engineered_data = os.path.join(base_path, r'engineered_data')

if not os.path.exists(path_engineered_data):
   os.makedirs(path_engineered_data)

##################################################
# open files

def read_and_modify_data(file_name, original_columns, year):
    file_path = os.path.join(base_path, file_name)
    df = pd.read_csv(file_path, encoding='utf-8')
    df_subset = df[df['시도교육청'].str.contains('서울특별시교육청')].copy()


    # generate boys/ratio info of each grade
    grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']

    for grade in grades:
        total_column = f"{grade}(합계)"
        columns_to_sum = [f"{grade}(남)", f"{grade}(여)"]
        df_subset.loc[:, total_column] = df_subset[columns_to_sum].sum(axis=1)

    grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']
    genders = ['남']

    for grade in grades:
        for gender in genders:
            column_name = f"{grade}({gender})"
            column_name_denominator = f"{grade}(합계)"
            df_subset[column_name + '비율'] = df_subset[column_name] / df_subset[column_name_denominator] * 100

    modified_columns = [column + year if column in original_columns else column for column in df_subset.columns]
    df_subset.columns = modified_columns

    return df_subset

# Define original columns and year
original_columns = ['1학년(남)', '1학년(여)', '2학년(남)', '2학년(여)', '3학년(남)', '3학년(여)',
                    '4학년(남)', '4학년(여)', '5학년(남)', '5학년(여)', '6학년(남)', '6학년(여)',
                    '특수학급(남)', '특수학급(여)', '순회학급(남)', '순회학급(여)', '계(남)', '계(여)', '총계',
                    '1학년(합계)', '2학년(합계)', '3학년(합계)', '4학년(합계)', '5학년(합계)', '6학년(합계)',
                    '1학년(남)비율', '2학년(남)비율', '3학년(남)비율', '4학년(남)비율', '5학년(남)비율', '6학년(남)비율']

year2022 = '2022'
year2021 = '2021'
year2020 = '2020'

# Read and modify 2022 data
file_name2022 = "data/elementarySchool/2022ElementarySchool.csv"
file_path = os.path.join(base_path, file_name2022)
ElementarySchoolStudentInfo2022_subset = read_and_modify_data(file_path, original_columns, year2022)
# print(ElementarySchoolStudentInfo2022_subset.columns)
# print(ElementarySchoolStudentInfo2022_subset['정보공시 학교코드'])

ElementarySchoolStudent2022 = 'data/elementarySchool/2022년도_학년별·학급별 학생수(초등).csv'
file_name = 'data/elementarySchool/2022년도_직위별 교원 현황.csv'
ElementarySchoolIncome2022 = 'data/elementarySchool/2022년도_학교회계 예·결산서(국공립)(예산)(수입).csv'
ElementarySchoolSpending2022 = 'data/elementarySchool/2022년도_학교회계 예·결산서(국공립)(예산)(지출).csv'
ElementarySchoolIncomingStudent2022 = 'data/elementarySchool/2022년도_전·출입 및 학업중단 학생 수(초등).csv'
ElementarySchoolOutgoingStudent2022 = 'data/elementarySchool/2022년도_전·출입 및 학업중단 학생 수(초등).csv'
ElementarySchoolOutgoingStudent2022 = 'data/elementarySchool/2022년도_입학생 현황.csv'

# Read the new files
ElementarySchoolStudent2022_df = pd.read_csv(os.path.join(base_path, ElementarySchoolStudent2022), encoding='utf-8')
ElementarySchoolTeacaherInfo2022_df = pd.read_csv(os.path.join(base_path, file_name), encoding='utf-8')
# ElementarySchoolIncome2022_df = pd.read_csv(os.path.join(base_path, ElementarySchoolIncome2022), encoding='utf-8')
# ElementarySchoolSpending2022_df = pd.read_csv(os.path.join(base_path, ElementarySchoolSpending2022), encoding='utf-8')
# ElementarySchoolIncomingStudent2022_df = pd.read_csv(os.path.join(base_path, ElementarySchoolIncomingStudent2022), encoding='utf-8')
# ElementarySchoolOutgoingStudent2022_df = pd.read_csv(os.path.join(base_path, ElementarySchoolOutgoingStudent2022), encoding='utf-8')

# ElementarySchoolTeacaherInfo2022_df_columns =  ElementarySchoolTeacaherInfo2022_df.columns
# for col in ElementarySchoolTeacaherInfo2022_df_columns:
#     print(col)

# ElementarySchoolStudentInfo2022_subset_columns =  ElementarySchoolStudentInfo2022_subset.columns
# for col in ElementarySchoolStudentInfo2022_subset_columns:
#     print(col)

# List of columns to exclude in the merge
exclude_columns = ['시도교육청', '지역교육청', '지역', '학교명', '학교급코드', '설립구분', '제외여부', '제외사유']

# print(len(ElementarySchoolStudentInfo2022_subset.columns))
# print(len(ElementarySchoolTeacaherInfo2022_df.columns))

# Perform the merge
merged_df = ElementarySchoolStudentInfo2022_subset.merge(
    ElementarySchoolTeacaherInfo2022_df.drop(exclude_columns, axis=1),
    on='정보공시 학교코드',
    how='inner'  # You can change this to 'left', 'right', or 'outer' if needed
)

# print(len(merged_df.columns))


# print(merged_df.head())

# merged_df_columns =  merged_df.columns
# for col in merged_df_columns:
#     print(col)

# Filter columns that contain numeric data
numeric_columns = merged_df.select_dtypes(include=[np.number]).columns

# Filter the specific columns that should remain as floats
float_columns = ['1학년(남)비율2022', '2학년(남)비율2022', '3학년(남)비율2022', '4학년(남)비율2022', '5학년(남)비율2022', '6학년(남)비율2022']

# Exclude the float columns from the numeric columns
int_columns = [col for col in numeric_columns if col not in float_columns]

# Convert the specific columns to integers
merged_df[int_columns] = merged_df[int_columns].astype(int)

# The columns you want to remain as floats (ratios)
float_columns = ['1학년(남)비율2022', '2학년(남)비율2022', '3학년(남)비율2022', '4학년(남)비율2022', '5학년(남)비율2022', '6학년(남)비율2022']

# Convert these columns to floats
merged_df[float_columns] = merged_df[float_columns].astype(float)

# print(merged_df.head())

ElementarySchoolStudent2022_df_columns = ElementarySchoolStudent2022_df.columns
# for col in ElementarySchoolStudent2022_df_columns:
#     print(col)
# print(ElementarySchoolStudent2022_df_columns)

# Perform the merge
ElementarySchoolStudentTeacher2022 = merged_df.merge(
    ElementarySchoolStudent2022_df.drop(exclude_columns, axis=1),
    on='정보공시 학교코드',
    how='inner'
)

# print(len(ElementarySchoolStudentTeacher2022.columns))

# pd.set_option('display.max_columns', None)
# print(ElementarySchoolStudentTeacher2022.columns)

ElementarySchoolStudentTeacher2022_columns = ElementarySchoolStudentTeacher2022.columns
for col in ElementarySchoolStudentTeacher2022_columns:
    print(col)

# # Specify the number of bins
# num_bins = 10
#
# # Specify the grades you want to create histograms for
# grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']
#
# # Create subplots for each grade
# plt.figure(figsize=(15, 10))
# for i, grade in enumerate(grades, 1):
#     plt.subplot(2, 3, i)  # 2 rows, 3 columns of subplots
#
#     # Calculate min and max values for the grade
#     min_val = ElementarySchoolStudentTeacher2022[f'{grade} 학급당 학생수'].min()
#     max_val = ElementarySchoolStudentTeacher2022[f'{grade} 학급당 학생수'].max()
#
#     # Create evenly spaced bins
#     bins = np.linspace(min_val, max_val, num_bins + 1)
#
#     # Create the histogram
#     plt.hist(ElementarySchoolStudentTeacher2022[f'{grade} 학급당 학생수'], bins=bins, edgecolor='k', alpha=0.7)
#     plt.xticks(bins)
#     plt.xlabel(f'Average Students per Class ({grade})')
#     plt.ylabel('Number of Schools')
#     plt.title(f'Distribution of Average Students per Class ({grade})')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()

# Specify the grades you want to create histograms for
grades = ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']

# # Specify the number of bins (common for all grades)
# num_bins = 10
#
# # Calculate the common range for all grades
# min_val = ElementarySchoolStudentTeacher2022[['1학년 학급당 학생수', '2학년 학급당 학생수', '3학년 학급당 학생수', '4학년 학급당 학생수', '5학년 학급당 학생수', '6학년 학급당 학생수']].min().min()
# max_val = ElementarySchoolStudentTeacher2022[['1학년 학급당 학생수', '2학년 학급당 학생수', '3학년 학급당 학생수', '4학년 학급당 학생수', '5학년 학급당 학생수', '6학년 학급당 학생수']].max().max()
#
# # Create subplots for each grade
# plt.figure(figsize=(15, 10))
# for i, grade in enumerate(grades, 1):
#     plt.subplot(2, 3, i)  # 2 rows, 3 columns of subplots
#
#     # Create evenly spaced bins based on the common range
#     bins = np.linspace(min_val, max_val, num_bins + 1)
#
#     # Create the histogram
#     plt.hist(ElementarySchoolStudentTeacher2022[f'{grade} 학급당 학생수'], bins=bins, edgecolor='k', alpha=0.7)
#     plt.xticks(bins)
#     plt.xlabel(f'Average Students per Class ({grade})')
#     plt.ylabel('Number of Schools')
#     plt.title(f'Distribution of Average Students per Class ({grade})')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()

# Calculate the mean and standard deviation for each grade
mean_values = []
std_values = []

for grade in grades:
    grade_column = ElementarySchoolStudentTeacher2022[f'{grade} 학급당 학생수']
    mean = grade_column.mean()
    std = grade_column.std()

    mean_values.append(mean)
    std_values.append(std)

for i, grade in enumerate(grades):
    print(f"{grade} - Mean: {mean_values[i]:.2f}, Standard Deviation: {std_values[i]:.2f}")

# Group the DataFrame by '지역' and calculate mean and standard deviation for each grade
mean_values = {grade: [] for grade in grades}
std_values = {grade: [] for grade in grades}

grouped = ElementarySchoolStudentTeacher2022.groupby('지역')

for grade in grades:
    for region, group in grouped:
        grade_column = group[f'{grade} 학급당 학생수']
        mean = grade_column.mean()
        std = grade_column.std()

        mean_values[grade].append(mean)
        std_values[grade].append(std)

for grade in grades:
    print(f"Grade {grade} - Mean Values by Region:")
    for region, mean in zip(grouped.groups.keys(), mean_values[grade]):
        print(f"{region}: {mean:.2f}")

    print(f"Grade {grade} - Standard Deviation by Region:")
    for region, std in zip(grouped.groups.keys(), std_values[grade]):
        print(f"{region}: {std:.2f}")


# # Create subplots for each grade
# plt.figure(figsize=(12, 6))
# for i, grade in enumerate(grades, 1):
#     plt.subplot(2, 3, i)  # 2 rows, 3 columns of subplots
#     sns.barplot(data=ElementarySchoolStudentTeacher2022, x='지역', y=f'{grade} 학급당 학생수', ci=None)
#     plt.title(f'Mean {grade} 학급당 학생수 by Region')
#     plt.xticks(rotation=45)
#     plt.xlabel('Region')
#     plt.ylabel('Mean')
#
# plt.tight_layout()
# plt.show()

from scipy import stats

# Define the grades you want to compare
grades = ['1학년 학급당 학생수', '2학년 학급당 학생수', '3학년 학급당 학생수', '4학년 학급당 학생수', '5학년 학급당 학생수', '6학년 학급당 학생수']

# List of unique locations (지역)
unique_locations = ElementarySchoolStudentTeacher2022['지역'].unique()

# Define the significance level (e.g., 0.05)
alpha = 0.05

for location in unique_locations:
    print(f"Location: {location}")
    for grade in grades:
        # Filter the DataFrame for the current location
        current_location = ElementarySchoolStudentTeacher2022[ElementarySchoolStudentTeacher2022['지역'] == location]

        # Filter the DataFrame for other locations (excluding the current location)
        other_locations = ElementarySchoolStudentTeacher2022[ElementarySchoolStudentTeacher2022['지역'] != location]

        # Perform a t-test to compare the current location to other locations for the selected grade
        t_stat, p_value = stats.ttest_ind(current_location[grade], other_locations[grade])

        # Check if the p-value is less than the significance level
        if p_value < alpha:
            print(f"{grade} in {location} is significantly different from other locations (p-value={p_value:.4f})")
        else:
            print(f"{grade} in {location} is not significantly different from other locations (p-value={p_value:.4f})")


