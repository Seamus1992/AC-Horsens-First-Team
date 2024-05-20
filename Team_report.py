import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt

def load_data():
    df_xg = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/xg_all DNK_1_Division_2023_2024.csv')
    df_xg['label'] = df_xg['label'] + ' ' + df_xg['date']

    df_xa = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/xA_all DNK_1_Division_2023_2024.csv')
    df_xa['label'] = df_xa['label'] + ' ' + df_xa['date']

    df_pv = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/pv_all DNK_1_Division_2023_2024.csv')
    df_pv['label'] = df_pv['label'] + ' ' + df_pv['date']

    df_possession = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/possession_stats_all DNK_1_Division_2023_2024.csv')
    df_possession['label'] = df_possession['label'] + ' ' + df_possession['date']

    return df_xg, df_xa, df_pv, df_possession

def generate_cumulative_chart(df, column, title, filename):
    plt.figure(figsize=(10, 5))
    for team in df['team_name'].unique():
        team_data = df[df['team_name'] == team]
        plt.plot(team_data['timeMin'] + team_data['timeSec'] / 60, team_data[column], label=team)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel(title)
    plt.title(f'Cumulative {title}')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Function to create a horizontal bar chart for expected points and probabilities
def create_bar_chart(value, title, filename, max_value, thresholds, annotations):
    fig, ax = plt.subplots(figsize=(6, 1))
    
    if value < thresholds[1]:
        bar_color = 'red'
    elif value < thresholds[2]:
        bar_color = 'yellow'
    else:
        bar_color = 'green'
    
    # Plot the full bar background
    ax.barh(0, max_value, color='lightgrey', height=0.3)
    
    # Plot the value bar with the determined color
    ax.barh(0, value, color=bar_color, height=0.3)
    # Plot the thresholds with annotations
    for threshold, color, annotation in zip(thresholds, ['red', 'yellow', 'green'], annotations):
        ax.axvline(threshold, color=color, linestyle='--', linewidth=1.5)
        ax.text(threshold, 0.45, f"{threshold:.2f}", ha='center', va='center', fontsize=8, color=color)
        ax.text(threshold, -0.5, annotation, ha='center', va='center', fontsize=8, color=color)
    
    # Add the text for title and value
    ax.text(max_value, 0.35, title, ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(value, -0.3, f"{value:.2f}", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
        
    # Formatting
    ax.set_xlim(0, max_value)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # Use high DPI for better quality
    plt.close()

# Function to simulate goals based on xG or xA values of each shot
def simulate_goals(values, num_simulations=10000):
    return np.random.binomial(1, values[:, np.newaxis], (len(values), num_simulations)).sum(axis=0)

# Function to simulate match outcomes based on xG or xA values of each shot
def simulate_match(home_values, away_values, num_simulations=10000):
    home_goals_simulated = simulate_goals(home_values, num_simulations)
    away_goals_simulated = simulate_goals(away_values, num_simulations)
    
    home_wins = np.sum(home_goals_simulated > away_goals_simulated)
    draws = np.sum(home_goals_simulated == away_goals_simulated)
    away_wins = np.sum(home_goals_simulated < away_goals_simulated)
    
    home_points = (home_wins * 3 + draws) / num_simulations
    away_points = (away_wins * 3 + draws) / num_simulations
    
    home_win_prob = home_wins / num_simulations
    draw_prob = draws / num_simulations
    away_win_prob = away_wins / num_simulations
    
    return home_points, away_points, home_win_prob, draw_prob, away_win_prob

# Function to calculate expected points and probabilities
def calculate_expected_points(df, value_column):
    expected_points_list = []
    total_expected_points = {team: 0 for team in df['team_name'].unique()}
    
    matches = df.groupby('label')
    for label, match_df in matches:
        teams = match_df['team_name'].unique()
        if len(teams) == 2:
            home_team, away_team = teams
            home_values = match_df[match_df['team_name'] == home_team][value_column].values
            away_values = match_df[match_df['team_name'] == away_team][value_column].values
            
            home_points, away_points, home_win_prob, draw_prob, away_win_prob = simulate_match(home_values, away_values)
            
            expected_points_list.append({
                'label': label, 
                'team_name': home_team, 
                'expected_points': home_points, 
                'win_probability': home_win_prob, 
                'draw_probability': draw_prob, 
                'loss_probability': away_win_prob
            })
            expected_points_list.append({
                'label': label, 
                'team_name': away_team, 
                'expected_points': away_points, 
                'win_probability': away_win_prob, 
                'draw_probability': draw_prob, 
                'loss_probability': home_win_prob
            })
            
            total_expected_points[home_team] += home_points
            total_expected_points[away_team] += away_points
    
    expected_points_df = pd.DataFrame(expected_points_list)
    total_expected_points_df = pd.DataFrame(list(total_expected_points.items()), columns=['team_name', 'total_expected_points'])
    total_expected_points_df = total_expected_points_df.sort_values(by='total_expected_points', ascending=False)
    
    return expected_points_df, total_expected_points_df

def preprocess_data(df_xg, df_xa, df_pv, df_possession):
    df_xg['timeMin'] = df_xg['timeMin'].astype(int)
    df_xg['timeSec'] = df_xg['timeSec'].astype(int)
    df_xg = df_xg.sort_values(by=['timeMin', 'timeSec'])
    df_xg['culmulativxg'] = df_xg.groupby('team_name')['321'].cumsum()

    df_xa['timeMin'] = df_xa['timeMin'].astype(int)
    df_xa['timeSec'] = df_xa['timeSec'].astype(int)
    df_xa = df_xa.sort_values(by=['timeMin', 'timeSec'])
    df_xa['culmulativxa'] = df_xa.groupby('team_name')['318.0'].cumsum()

    df_pv['timeMin'] = df_pv['timeMin'].astype(int)
    df_pv['timeSec'] = df_pv['timeSec'].astype(int)
    df_pv = df_pv.sort_values(by=['timeMin', 'timeSec'])
    df_pv['culmulativpv'] = df_pv.groupby('team_name')['possessionValue.pvValue'].cumsum()

    return df_xg, df_xa, df_pv, df_possession

df_xg, df_xa, df_pv, df_possession = load_data()

df_xg_agg, df_xa_agg, df_pv_agg, df_possession = preprocess_data(df_xg, df_xa, df_pv, df_possession)

# Calculate expected points based on xG
expected_points_xg, total_expected_points_xg = calculate_expected_points(df_xg, '321')

# Calculate expected points based on xA
expected_points_xa, total_expected_points_xa = calculate_expected_points(df_xa, '318.0')

# Merge the expected points from both xG and xA simulations
merged_df = expected_points_xg.merge(expected_points_xa, on=['label', 'team_name'], suffixes=('_xg', '_xa'))
merged_df['expected_points'] = (merged_df['expected_points_xg'] + merged_df['expected_points_xa']) / 2
merged_df['win_probability'] = (merged_df['win_probability_xg'] + merged_df['win_probability_xa']) / 2
merged_df['draw_probability'] = (merged_df['draw_probability_xg'] + merged_df['draw_probability_xa']) / 2
merged_df['loss_probability'] = (merged_df['loss_probability_xg'] + merged_df['loss_probability_xa']) / 2

# Filter the data for Horsens
horsens_df = merged_df[merged_df['team_name'] == 'Horsens']

# Function to create a PDF report for each game
def create_pdf_report(game_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    label = game_data['label']
    expected_points = game_data['expected_points']
    win_prob = game_data['win_probability']
    
    # Add the team logo
    pdf.image('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/Logo.png', x=165, y=10, w=15, h=15)
    pdf.set_xy(10, 10)
    pdf.cell(140, 10, txt=f"Match Report: {label}", ln=True, align='L')
    
    pdf.ln(10)
    
    # Create bar charts for expected points and probabilities
    create_bar_chart(expected_points, 'Expected Points', 'bar_combined.png', 3.0, [1.0, 1.2, 1.8], ['Relegation','Top 6', 'Promotion'])
    create_bar_chart(win_prob, 'Win Probability', 'bar_combined_win_prob.png', 1.0, [0.2, 0.4, 0.6], ['Low', 'Medium', 'High'])
    
    # Add bar charts to PDF side by side
    pdf.image('bar_combined.png', x=10, y=30, w=90, h=20)
    pdf.image('bar_combined_win_prob.png', x=110, y=30, w=90, h=20)
        
    pdf.output(f"Match reports/Horsens_Report_{label}.pdf")

# Generate a PDF report for each game involving Horsens
for index, row in horsens_df.iterrows():
    create_pdf_report(row)

# Calculate total combined expected points for Horsens
total_expected_points_combined = total_expected_points_xg.merge(total_expected_points_xa, on='team_name', suffixes=('_xg', '_xa'))
total_expected_points_combined['total_expected_points'] = (total_expected_points_combined['total_expected_points_xg'] + total_expected_points_combined['total_expected_points_xa']) / 2
total_expected_points_combined = total_expected_points_combined[['team_name', 'total_expected_points']]

# Print the total combined expected points for Horsens
print("\nTotal Combined Expected Points for Horsens:")
print(total_expected_points_combined)
