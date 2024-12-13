{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imports\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_datas():\n",
    "        general_data = pd.read_csv(\"datas\\\\general_data.csv\")\n",
    "        employee_data = pd.read_csv(\"datas\\\\employee_survey_data.csv\")\n",
    "        manager_data = pd.read_csv(\"datas\\\\manager_survey_data.csv\")\n",
    "\n",
    "        general_data, employee_data, manager_data = load_datas()\n",
    "        print(general_data.describe())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
