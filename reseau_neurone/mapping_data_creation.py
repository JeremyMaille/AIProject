import pandas as pd
import numpy as np

def generate_mapping_values(input_file, output_file):
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()

    # Numerical columns with their ranges
    numerical_columns = [
        'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
        'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement',
        'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
        'WorkLifeBalance', 'AverageHoursPerDay', 'IrregularDays'
    ]

    # Create mappings dictionary
    mappings = {}

    # Map numerical columns with actual ranges
    for col in numerical_columns:
        if col in data.columns:
            min_val = data[col].min()
            max_val = data[col].max()
            mappings[col] = f'"[{min_val}, {max_val}]"'

    # Define categorical columns with individual mappings
    binary_columns = {
        'BusinessTravel_Travel_Frequently': '"[-1, 1]"',
        'BusinessTravel_Travel_Rarely': '"[-1, 1]"',
        'Department_Research & Development': '"[-1, 1]"',
        'Department_Sales': '"[-1, 1]"',
        'EducationField_Life Sciences': '"[-1, 1]"',
        'EducationField_Marketing': '"[-1, 1]"',
        'EducationField_Medical': '"[-1, 1]"',
        'EducationField_Other': '"[-1, 1]"',
        'EducationField_Technical Degree': '"[-1, 1]"',
        'Gender_Male': '"[-1, 1]"',
        'JobRole_Human Resources': '"[-1, 1]"',
        'JobRole_Laboratory Technician': '"[-1, 1]"',
        'JobRole_Manager': '"[-1, 1]"',
        'JobRole_Manufacturing Director': '"[-1, 1]"',
        'JobRole_Research Director': '"[-1, 1]"',
        'JobRole_Research Scientist': '"[-1, 1]"',
        'JobRole_Sales Executive': '"[-1, 1]"',
        'JobRole_Sales Representative': '"[-1, 1]"',
        'MaritalStatus_Married': '"[-1, 1]"',
        'MaritalStatus_Single': '"[-1, 1]"'
    }

    # Add categorical mappings
    mappings.update(binary_columns)

    # Write to CSV maintaining column order
    columns = numerical_columns + list(binary_columns.keys())
    header_row = ','.join(columns)
    mapping_row = ','.join(mappings[col] for col in columns)

    with open(output_file, 'w') as f:
        f.write(header_row + '\n')
        f.write(mapping_row + '\n')

# Generate mapping values
generate_mapping_values(
    input_file='datas/final_merged_data_with_work_metrics_delete.csv',
    output_file='mapping_values.csv'
)