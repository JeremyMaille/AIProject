# ------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ------------------------------------------
import pandas as pd
import numpy as np
import pygame
import sys
from tensorflow.keras.models import load_model
import joblib

# ------------------------------------------
# LOADING MODEL, SCALER, AND DATA
# ------------------------------------------
model = load_model('employee_attrition_model.h5')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('datas\\FINAL_MERGED_DATA.CSV')
data.columns = data.columns.str.strip()
data = pd.get_dummies(data, columns=[
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Gender_Male',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'Department_Research & Development',
    'Department_Sales', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative'
])
X = data.drop(['Attrition', 'EmployeeID'], axis=1)

# ------------------------------------------
# INITIALIZING PYGAME
# ------------------------------------------
pygame.init()

# ------------------------------------------
# CONFIGURING DISPLAY
# ------------------------------------------
width, height = 1600, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Prédiction de départ des employés')

# ------------------------------------------
# DEFINING COLORS AND FONT
# ------------------------------------------
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')
COLOR_TEXT = pygame.Color('black')
COLOR_BG = pygame.Color('white')
FONT = pygame.font.Font(None, 24)
FONT_TITLE = pygame.font.Font(None, 32)

# ------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------
class InputBox:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = ''
        self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)
        self.active = False
        self.label = label

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    pass
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        label_surface = FONT.render(self.label, True, COLOR_TEXT)
        screen.blit(label_surface, (self.rect.x - 250, self.rect.y + 5))

class Dropdown:
    def __init__(self, x, y, w, h, label, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = ''
        self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)
        self.active = False
        self.label = label
        self.options = options
        self.selected_option = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        label_surface = FONT.render(self.label, True, COLOR_TEXT)
        screen.blit(label_surface, (self.rect.x - 250, self.rect.y + 5))
        if self.active:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i+1) * self.rect.height, self.rect.width, self.rect.height)
                pygame.draw.rect(screen, self.color, option_rect, 2)
                option_surface = FONT.render(option, True, COLOR_TEXT)
                screen.blit(option_surface, (option_rect.x+5, option_rect.y+5))

class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_ACTIVE
        self.text = text
        self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_rect = self.txt_surface.get_rect(center=self.rect.center)
        screen.blit(self.txt_surface, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# ------------------------------------------
# SETTING UP INPUTS AND DROPDOWNS
# ------------------------------------------
numeric_features = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
categorical_features = ['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Gender_Male', 'MaritalStatus_Married', 'MaritalStatus_Single', 'Department_Research & Development', 'Department_Sales', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative']

input_boxes = []
dropdowns = []
start_x = width // 2 - 200
start_y = 100
box_width = 140
box_height = 32
padding = 10

half = len(numeric_features) // 2 + len(numeric_features) % 2
for i, feature in enumerate(numeric_features):
    x = start_x if i < half else start_x + 400
    y = start_y + (i % half) * (box_height + padding)
    input_box = InputBox(x, y, box_width, box_height, feature)
    input_boxes.append(input_box)

dropdown_options = {
    'BusinessTravel': ['Travel_Frequently', 'Travel_Rarely'],
    'Gender': ['Male', 'Female'],
    'MaritalStatus': ['Married', 'Single'],
    'Department': ['Research & Development', 'Sales'],
    'JobRole': ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
}

for i, feature in enumerate(categorical_features):
    x = start_x if i < half else start_x + 400
    y = start_y + (i % half + len(numeric_features)) * (box_height + padding)
    dropdown = Dropdown(x, y, box_width, box_height, feature, dropdown_options[feature.split('_')[0]])
    dropdowns.append(dropdown)

predict_button = Button(width // 2 - 50, height - 100, 100, 40, 'Prédire')

# ------------------------------------------
# MAIN LOOP
# ------------------------------------------
prediction_text = ''
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        for box in input_boxes:
            box.handle_event(event)

        for dropdown in dropdowns:
            dropdown.handle_event(event)

        if predict_button.is_clicked(event):
            input_values = []
            for box in input_boxes:
                try:
                    value = float(box.text)
                except ValueError:
                    value = 0.0
                input_values.append(value)
            for dropdown in dropdowns:
                input_values.append(1 if dropdown.selected_option == 'True' else 0)
            
            input_df = pd.DataFrame([input_values], columns=X.columns)
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_data = scaler.transform(input_df)
            prediction = model.predict(input_data)
            probability = prediction[0][0] * 100
            prediction_text = f"Probabilité de départ : {probability:.2f}%"

    screen.fill(COLOR_BG)

    title_surface = FONT_TITLE.render('Prédiction de départ des employés', True, COLOR_TEXT)
    title_rect = title_surface.get_rect(center=(width // 2, 30))
    screen.blit(title_surface, title_rect)

    for box in input_boxes:
        box.draw(screen)

    for dropdown in dropdowns:
        dropdown.draw(screen)

    predict_button.draw(screen)

    if prediction_text:
        prediction_surface = FONT.render(prediction_text, True, COLOR_TEXT)
        prediction_rect = prediction_surface.get_rect(center=(width // 2, height - 50))
        screen.blit(prediction_surface, prediction_rect)

    pygame.display.flip()

pygame.quit()
