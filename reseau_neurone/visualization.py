import pandas as pd
import numpy as np
import pygame
import sys
from tensorflow.keras.models import load_model
import joblib

# Charger le modèle entraîné et le scaler
model = load_model('models\\best_model_1_100.00.keras')
scaler = joblib.load('scaler.pkl')

# Charger les données pour obtenir les colonnes
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

# Initialiser Pygame
pygame.init()

# Configurer l'affichage
width, height = 1600, 900  # Augmenter la taille de la fenêtre
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Prédiction de départ des employés')

# Définir les couleurs
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')
COLOR_TEXT = pygame.Color('black')
COLOR_BG = pygame.Color('white')

# Définir la police
FONT = pygame.font.Font(None, 24)
FONT_TITLE = pygame.font.Font(None, 32)

# Classe InputBox pour les champs de saisie
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
            # Si l'utilisateur clique sur la boîte
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            # Changer la couleur
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    pass
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render le texte
                self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)

    def draw(self, screen):
        # Dessiner le rectangle
        pygame.draw.rect(screen, self.color, self.rect, 2)
        # Afficher le texte
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Afficher le label
        label_surface = FONT.render(self.label, True, COLOR_TEXT)
        screen.blit(label_surface, (self.rect.x - 250, self.rect.y + 5))  # Ajuster la position du label

# Classe Dropdown pour les listes déroulantes
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
            # Si l'utilisateur clique sur la boîte
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            # Changer la couleur
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    pass
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render le texte
                self.txt_surface = FONT.render(self.text, True, COLOR_TEXT)

    def draw(self, screen):
        # Dessiner le rectangle
        pygame.draw.rect(screen, self.color, self.rect, 2)
        # Afficher le texte
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Afficher le label
        label_surface = FONT.render(self.label, True, COLOR_TEXT)
        screen.blit(label_surface, (self.rect.x - 250, self.rect.y + 5))  # Ajuster la position du label
        # Afficher les options si actif
        if self.active:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i+1) * self.rect.height, self.rect.width, self.rect.height)
                pygame.draw.rect(screen, self.color, option_rect, 2)
                option_surface = FONT.render(option, True, COLOR_TEXT)
                screen.blit(option_surface, (option_rect.x+5, option_rect.y+5))

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# Classe Button pour le bouton Prédire
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

# Créer des InputBox pour chaque caractéristique
numeric_features = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
categorical_features = ['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Gender_Male', 'MaritalStatus_Married', 'MaritalStatus_Single', 'Department_Research & Development', 'Department_Sales', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative']

input_boxes = []
dropdowns = []
start_x = width // 2 - 200  # Centrer horizontalement
start_y = 100
box_width = 140
box_height = 32
padding = 10

# Calculer le placement pour deux colonnes
half = len(numeric_features) // 2 + len(numeric_features) % 2
for i, feature in enumerate(numeric_features):
    if i < half:
        x = start_x
        y = start_y + i * (box_height + padding)
    else:
        x = start_x + 400
        y = start_y + (i - half) * (box_height + padding)
    input_box = InputBox(x, y, box_width, box_height, feature)
    input_boxes.append(input_box)

# Ajouter les Dropdown pour les caractéristiques catégorielles
dropdown_options = {
    'BusinessTravel': ['Travel_Frequently', 'Travel_Rarely'],
    'Gender': ['Male', 'Female'],
    'MaritalStatus': ['Married', 'Single'],
    'Department': ['Research & Development', 'Sales'],
    'JobRole': ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
}

for i, feature in enumerate(categorical_features):
    if i < half:
        x = start_x
        y = start_y + (i + len(numeric_features)) * (box_height + padding)
    else:
        x = start_x + 400
        y = start_y + ((i + len(numeric_features)) - half) * (box_height + padding)
    dropdown = Dropdown(x, y, box_width, box_height, feature, dropdown_options[feature.split('_')[0]])
    dropdowns.append(dropdown)

# Créer le bouton Prédire
predict_button = Button(width // 2 - 50, height - 100, 100, 40, 'Prédire')

# Texte de prédiction
prediction_text = ''

# Boucle principale
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
            # Récupérer les valeurs saisies
            input_values = []
            for box in input_boxes:
                try:
                    value = float(box.text)
                except ValueError:
                    value = 0.0  # Valeur par défaut si la saisie échoue
                input_values.append(value)
            for dropdown in dropdowns:
                if dropdown.selected_option:
                    input_values.append(1 if dropdown.selected_option == 'True' else 0)
                else:
                    input_values.append(0)  # Valeur par défaut si aucune option n'est sélectionnée
            
            # Créer un DataFrame avec les valeurs saisies
            input_df = pd.DataFrame([input_values], columns=X.columns)
            
            # Remplir les colonnes manquantes avec des zéros
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Normaliser les entrées
            input_data = scaler.transform(input_df)
            
            # Faire la prédiction
            prediction = model.predict(input_data)
            
            # Calculer le pourcentage
            probability = prediction[0][0] * 100
            
            # Mettre à jour le texte de prédiction avec pourcentage
            prediction_text = f"Probabilité de départ : {probability:.2f}%"

    # Remplir l'écran
    screen.fill(COLOR_BG)

    # Afficher le titre
    title_surface = FONT_TITLE.render('Prédiction de départ des employés', True, COLOR_TEXT)
    title_rect = title_surface.get_rect(center=(width // 2, 30))
    screen.blit(title_surface, title_rect)

    # Dessiner les InputBox
    for box in input_boxes:
        box.draw(screen)

    # Dessiner les Dropdown
    for dropdown in dropdowns:
        dropdown.draw(screen)

    # Dessiner le bouton Prédire
    predict_button.draw(screen)

    # Afficher le résultat de la prédiction
    if prediction_text:
        prediction_surface = FONT.render(prediction_text, True, COLOR_TEXT)
        prediction_rect = prediction_surface.get_rect(center=(width // 2, height - 50))
        screen.blit(prediction_surface, prediction_rect)

    pygame.display.flip()

pygame.quit()