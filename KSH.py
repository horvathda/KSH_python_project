"""
Gábor Dénes Egyetem Programozási alapok tantárgy beadandó dolgozat Zero to Hero csoport programkódja.
Készítette: XY
Tesztelte: XY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# 1. lépés: A CSV fájl betöltése és előkészítése
file_path = 'stadat-jov0045-14.1.2.4-hu.csv'
try:
    data = pd.read_csv(file_path, encoding='cp1250', sep=';', skiprows=1)
except UnicodeDecodeError as e:
    messagebox.showerror("Hiba", "Nem sikerült beolvasni a CSV fájlt. Ellenőrizze a fájl kódolását és formátumát.")
    raise e

# Oszlopok tisztítása és átnevezése
data.columns = [col.strip() for col in data.columns]
data.rename(columns={'Megnevezés': 'Jövedelem_típus', 'Ország összesen': 'Orszag_osszesen'}, inplace=True)

# Az év oszlop hozzáadása és jövedelem típus szerinti szűrés
melted_data = pd.DataFrame()
current_year = None
rows = []

for i, row in data.iterrows():
    if str(row.iloc[0]).isdigit():  # Ha a sor évszámot tartalmaz
        current_year = row.iloc[0]
    else:  # Egyéb esetben jövedelem adatok
        if current_year:
            row['Év'] = current_year
            rows.append(row)

melted_data = pd.DataFrame(rows)
melted_data = melted_data[melted_data['Jövedelem_típus'].isin(['Bruttó jövedelem', 'Nettó jövedelem'])]
melted_data['Év'] = pd.to_numeric(melted_data['Év'], errors='coerce')
melted_data.dropna(subset=['Év'], inplace=True)

# Numerikus konverzió a régiós és település típusú oszlopokra
for col in melted_data.columns[1:-1]:
    melted_data[col] = melted_data[col].replace({' ': '', '…': '0'}, regex=True).str.replace(',', '').astype(float)
melted_data.dropna(inplace=True)

# Formázó függvény a forint megjelenítéshez
def forint_formatter(x, pos):
    return f'{int(x):,} Ft'

# Diagramok megjelenítése funkciók
def show_country_line_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    for jovedelem_tipus in melted_data['Jövedelem_típus'].unique():
        subset = melted_data[melted_data['Jövedelem_típus'] == jovedelem_tipus]
        ax.plot(subset['Év'], subset['Orszag_osszesen'], marker='o', linestyle='-', label=jovedelem_tipus)
    ax.set_xlabel('Év')
    ax.set_ylabel('Ország összesen')
    ax.set_title('Ország összes jövedelem (Vonal Diagram)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_country_scatter_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(ax=ax, x=melted_data['Év'], y=melted_data['Orszag_osszesen'], hue=melted_data['Jövedelem_típus'])
    ax.set_xlabel('Év')
    ax.set_ylabel('Ország összesen')
    ax.set_title('Ország összes jövedelem (Szórási Diagram)')
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_country_regression():
    brutto_data = melted_data[melted_data['Jövedelem_típus'] == 'Bruttó jövedelem']
    X = brutto_data[['Év']].values.reshape(-1, 1)
    y = brutto_data['Orszag_osszesen'].values.reshape(-1, 1)

    if X.size == 0 or y.size == 0 or X.shape[0] != y.shape[0]:
        messagebox.showerror("Hiba", "Nem találhatók adatok a regresszióhoz.")
        return

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.6, label='Valós adatok')
    ax.plot(X, y_pred, color='red', linestyle='--', label='Lineáris regresszió')
    ax.set_xlabel('Év')
    ax.set_ylabel('Ország összesen')
    ax.set_title('Ország összes jövedelem (Lineáris regresszió)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

# Régiók szerinti diagramok
def show_region_line_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in [col for col in melted_data.columns if 'Régiók szerint' in col]:
        ax.plot(melted_data['Év'], melted_data[region], marker='o', linestyle='-', label=region)
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem régiónként')
    ax.set_title('Jövedelem alakulása régiónként (Vonal Diagram)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_region_scatter_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in [col for col in melted_data.columns if 'Régiók szerint' in col]:
        sns.scatterplot(ax=ax, x=melted_data['Év'], y=melted_data[region], label=region)
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem régiónként')
    ax.set_title('Jövedelem alakulása régiónként (Szórási Diagram)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_region_regression(region):
    X = melted_data[['Év']].values.reshape(-1, 1)
    y = melted_data[region].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label=f'Valós adatok - {region}')
    ax.plot(X, y_pred, linestyle='--', label=f'Regresszió - {region}')
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem')
    ax.set_title(f'Lineáris regresszió - {region}')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

# Településtípus szerinti diagramok
def show_settlement_type_line_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    for settlement_type in [col for col in melted_data.columns if 'Települések típusa szerint' in col]:
        ax.plot(melted_data['Év'], melted_data[settlement_type], marker='o', linestyle='-', label=settlement_type)
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem településtípus szerint')
    ax.set_title('Jövedelem településtípus szerint (Vonal Diagram)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_settlement_type_scatter_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    for settlement_type in [col for col in melted_data.columns if 'Települések típusa szerint' in col]:
        sns.scatterplot(ax=ax, x=melted_data['Év'], y=melted_data[settlement_type], label=settlement_type)
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem településtípus szerint')
    ax.set_title('Jövedelem településtípus szerint (Szórási Diagram)')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

def show_settlement_type_regression(settlement_type):
    X = melted_data[['Év']].values.reshape(-1, 1)
    y = melted_data[settlement_type].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label=f'Valós adatok - {settlement_type}')
    ax.plot(X, y_pred, linestyle='--', label=f'Regresszió - {settlement_type}')
    ax.set_xlabel('Év')
    ax.set_ylabel('Jövedelem')
    ax.set_title(f'Lineáris regresszió - {settlement_type}')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(forint_formatter))
    plt.show()

# Tkinter ablak létrehozása
root = tk.Tk()
root.title("Az egy főre jutó bruttó és nettó jövedelem régió és településtípus szerint")

# Notebook widget a lapfülekhez
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Országos jövedelem lap
country_tab = ttk.Frame(notebook)
notebook.add(country_tab, text="Országos jövedelem")
tk.Button(country_tab, text="Vonal Diagram", command=show_country_line_chart).pack(pady=5)
tk.Button(country_tab, text="Szórási Diagram", command=show_country_scatter_plot).pack(pady=5)
tk.Button(country_tab, text="Regresszió", command=show_country_regression).pack(pady=5)

# Régiók szerinti jövedelem lap
region_tab = ttk.Frame(notebook)
notebook.add(region_tab, text="Régiók szerinti jövedelem")
tk.Button(region_tab, text="Vonal Diagram", command=show_region_line_chart).pack(pady=5)
tk.Button(region_tab, text="Szórási Diagram", command=show_region_scatter_plot).pack(pady=5)

# Külön regressziós gombok minden régióhoz
for region in [col for col in melted_data.columns if 'Régiók szerint' in col]:
    tk.Button(region_tab, text=f"Regresszió - {region}", command=lambda r=region: show_region_regression(r)).pack(pady=5)

# Településtípus szerinti jövedelem lap
settlement_tab = ttk.Frame(notebook)
notebook.add(settlement_tab, text="Településtípusok szerinti jövedelem")
tk.Button(settlement_tab, text="Vonal Diagram", command=show_settlement_type_line_chart).pack(pady=5)
tk.Button(settlement_tab, text="Szórási Diagram", command=show_settlement_type_scatter_plot).pack(pady=5)

# Külön regressziós gombok minden településtípushoz
for settlement_type in [col for col in melted_data.columns if 'Települések típusa szerint' in col]:
    tk.Button(settlement_tab, text=f"Regresszió - {settlement_type}", command=lambda st=settlement_type: show_settlement_type_regression(st)).pack(pady=5)

root.mainloop()
