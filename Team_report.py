import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

def load_data():
    df_xg = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/xg_all DNK_1_Division_2023_2024.csv')
    df_xg['label'] = df_xg['label'] + ' ' + df_xg['date']

    df_xa = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/xA_all DNK_1_Division_2023_2024.csv')
    df_xa['label'] = df_xa['label'] + ' ' + df_xa['date']

    df_pv = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/pv_all DNK_1_Division_2023_2024.csv')

    df_possession_stats = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/possession_stats_all DNK_1_Division_2023_2024.csv')
    df_possession_stats['label'] = df_possession_stats['label'] + ' ' + df_possession_stats['date']

    df_xa_agg = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/Horsens/Horsens_possession_data.csv')
    df_xa_agg['label'] = df_xa_agg['label'] + ' ' + df_xa_agg['date']

    df_xg_agg = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/Horsens/Horsens_xg_data.csv')
    df_xg_agg['label'] = df_xg_agg['label'] + ' ' + df_xg_agg['date']

    df_pv_agg = pd.read_csv('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2023_2024/Horsens/Horsens_pv_data.csv')
    df_pv_agg['label'] = df_pv_agg['label'] + ' ' + df_pv_agg['date']

    return df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_xg_agg, df_pv_agg

def create_bar_chart(value, title, filename, max_value, thresholds, annotations):
    fig, ax = plt.subplots(figsize=(8, 2))
    
    if value < thresholds[0]:
        bar_color = 'red'
    elif value < thresholds[1]:
        bar_color = 'orange'
    elif value < thresholds[2]:
        bar_color = 'yellow'
    else:
        bar_color = 'green'
    
    # Plot the full bar background
    ax.barh(0, max_value, color='lightgrey', height=0.5)
    
    # Plot the value bar with the determined color
    ax.barh(0, value, color=bar_color, height=0.5)
    # Plot the thresholds with annotations
    for threshold, color, annotation in zip(thresholds, ['red', 'yellow', 'green'], annotations):
        ax.axvline(threshold, color=color, linestyle='--', linewidth=1.5)
        ax.text(threshold, 0.3, f"{threshold:.2f}", ha='center', va='center', fontsize=10, color=color)
        ax.text(threshold, -0.3, annotation, ha='center', va='center', fontsize=10, color=color)
    
    # Add the text for title and value
    ax.text(value, -0.0, f"{value:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        
    # Formatting
    ax.set_xlim(0, max_value)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    plt.title(label=title, fontsize=12, fontweight='bold',y=1.2,va='top', loc='left')   
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # Use high DPI for better quality
    plt.close()

def generate_cumulative_chart(df, column, title, filename):
    plt.figure(figsize=(12, 6))
    for team in df['team_name'].unique():
        team_data = df[df['team_name'] == team]
        plt.plot(team_data['timeMin'] + team_data['timeSec'] / 60, team_data[column], label=team)
    
    plt.xlabel('Time (minutes)')
    plt.title(f'Cumulative {title}')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def generate_possession_chart(df_possession_stats, filename):
    cols_to_average = df_possession_stats.columns[[6, 7, 8]]
    df_possession_stats[cols_to_average] = df_possession_stats[cols_to_average].apply(pd.to_numeric, errors='coerce')
    time_column = 'interval'

    # Calculate sliding average
    sliding_average = df_possession_stats[cols_to_average].rolling(window=2).mean()
    # Plotting
    plt.figure(figsize=(12,6))
    sorted_columns = sliding_average.columns.sort_values()  # Sort the columns alphabetically
    for col in sorted_columns:
        plt.plot(df_possession_stats[time_column], sliding_average[col], label=col)

    plt.xlabel('Time (minutes)')
    plt.title('Sliding average territorial possession')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def simulate_goals(values, num_simulations=10000):
    return np.random.binomial(1, values[:, np.newaxis], (len(values), num_simulations)).sum(axis=0)

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

def preprocess_data(df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats):
    df_xg_agg['timeMin'] = df_xg_agg['timeMin'].astype(int)
    df_xg_agg['timeSec'] = df_xg_agg['timeSec'].astype(int)
    df_xg_agg = df_xg_agg.sort_values(by=['team_name','timeMin', 'timeSec'])
    df_xg_agg['cumulativxg'] = df_xg_agg.groupby(['team_name','label'])['321'].cumsum()

    df_xa_agg['timeMin'] = df_xa_agg['timeMin'].astype(int)
    df_xa_agg['timeSec'] = df_xa_agg['timeSec'].astype(int)
    df_xa_agg = df_xa_agg.sort_values(by=['team_name','timeMin', 'timeSec'])
    df_xa_agg = df_xa_agg[df_xa_agg['318.0'].astype(float) > 0]

    df_xa_agg['cumulativxa'] = df_xa_agg.groupby(['team_name', 'label'])['318.0'].cumsum()
    
    df_possession_stats = df_possession_stats[df_possession_stats['type'] == 'territorialThird'].copy()
    df_possession_stats.loc[:, 'home'] = df_possession_stats['home'].astype(float)
    df_possession_stats.loc[:, 'away'] = df_possession_stats['away'].astype(float)

    df_possession_stats_summary = df_possession_stats.groupby(['home_team', 'away_team', 'label']).agg({'home': 'mean', 'away': 'mean'}).reset_index()
    df_possession_stats_summary = df_possession_stats_summary.rename(columns={'home': 'home_possession', 'away': 'away_possession'})

    df_possession_stats = df_possession_stats.drop_duplicates()
    df_possession_stats = df_possession_stats[df_possession_stats['interval_type'] == 5]

    return df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats, df_possession_stats_summary

def create_holdsummary(df_possession_stats_summary, df_xg, df_xa):
    df_possession_stats_summary = pd.melt(df_possession_stats_summary, id_vars=['home_team', 'away_team', 'label'], value_vars=['home_possession', 'away_possession'], var_name='possession_type', value_name='terr_poss')
    df_possession_stats_summary['team_name'] = df_possession_stats_summary.apply(lambda x: x['home_team'] if x['possession_type'] == 'home_possession' else x['away_team'], axis=1)
    df_possession_stats_summary.drop(['home_team', 'away_team', 'possession_type'], axis=1, inplace=True)
    df_possession_stats_summary = df_possession_stats_summary[['team_name', 'label', 'terr_poss']]
    df_xg_hold = df_xg.groupby(['team_name', 'label'])['321'].sum().reset_index()
    df_xg_hold = df_xg_hold.rename(columns={'321': 'xG'})

    df_xa_hold = df_xa.groupby(['team_name', 'label'])['318.0'].sum().reset_index()
    df_xa_hold = df_xa_hold.rename(columns={'318.0': 'xA'})

    df_holdsummary = df_xa_hold.merge(df_xg_hold)
    df_holdsummary = df_holdsummary.merge(df_possession_stats_summary)
    df_holdsummary = df_holdsummary[['team_name', 'label', 'xA', 'xG', 'terr_poss']]
    
    return df_holdsummary

df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_xg_agg, df_pv_agg = load_data()

df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats, df_possession_stats_summary = preprocess_data(df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats)

# Calculate expected points based on xG
expected_points_xg, total_expected_points_xg = calculate_expected_points(df_xg, '321')

# Calculate expected points based on xA
expected_points_xa, total_expected_points_xa = calculate_expected_points(df_xa, '318.0')

df_holdsummary = create_holdsummary(df_possession_stats_summary, df_xg, df_xa)
# Merge the expected points from both xG and xA simulations
merged_df = expected_points_xg.merge(expected_points_xa, on=['label', 'team_name'], suffixes=('_xg', '_xa'))
merged_df['expected_points'] = (merged_df['expected_points_xg'] + merged_df['expected_points_xa']) / 2
merged_df['win_probability'] = (merged_df['win_probability_xg'] + merged_df['win_probability_xa']) / 2
merged_df['draw_probability'] = (merged_df['draw_probability_xg'] + merged_df['draw_probability_xa']) / 2
merged_df['loss_probability'] = (merged_df['loss_probability_xg'] + merged_df['loss_probability_xa']) / 2
merged_df = merged_df.merge(df_holdsummary,on=['label', 'team_name'])
horsens_df = merged_df[merged_df['team_name'] == 'Horsens']
# Function to create a PDF report for each game
def create_pdf_report(game_data, df_xg_agg, df_xa_agg, merged_df, df_possession_stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    label = game_data['label']
    expected_points = game_data['expected_points']
    win_prob = game_data['win_probability']
    draw_prob = game_data['draw_probability']
    loss_prob = game_data['loss_probability']
    
    # Add the team logo
    pdf.image('C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/Logo.png', x=165, y=10, w=15, h=15)
    pdf.set_xy(10, 10)
    pdf.cell(140, 10, txt=f"Match Report: {label}", ln=True, align='L')
    
    pdf.ln(8)
    
    # Create bar charts for expected points and probabilities
    create_bar_chart(expected_points, 'Expected Points', 'bar_combined.png', 3.0, [1.0, 1.3, 1.8], ['Relegation','Top 6', 'Promotion'])
    create_bar_chart(win_prob, 'Win Probability', 'bar_combined_win_prob.png', 1.0, [0.2, 0.4, 0.6], ['Low', 'Medium', 'High'])
    
    # Add bar charts to PDF side by side
    pdf.image('bar_combined.png', x=10, y=30, w=90, h=30)
    pdf.image('bar_combined_win_prob.png', x=110, y=30, w=90, h=30)

    pdf.set_xy(10, 60)

    game_xg_agg = df_xg_agg[df_xg_agg['label'] == label]
    game_xa_agg = df_xa_agg[df_xa_agg['label'] == label]
    df_possession_stats = df_possession_stats[df_possession_stats['label'] == label]
    first_home_team = df_possession_stats['home_team'].iloc[0]
    first_away_team = df_possession_stats['away_team'].iloc[0]
    df_possession_stats = df_possession_stats.rename(columns={'home': first_home_team, 'away': first_away_team})
    generate_cumulative_chart(game_xg_agg, 'cumulativxg', 'xG', 'cumulative_xg.png')
    generate_cumulative_chart(game_xa_agg, 'cumulativxa', 'xA', 'cumulative_xa.png')
    generate_possession_chart(df_possession_stats, 'cumulative_possession.png')

    pdf.image('cumulative_xg.png', x=5, y=70, w=66, h=60)
    pdf.image('cumulative_xa.png', x=72, y=70, w=66, h=60)
    pdf.image('cumulative_possession.png', x=139, y=70, w=66, h=60)

    # Add a summary table
    pdf.set_xy(10, 130)
    pdf.set_font_size(10)
    pdf.cell(20, 10, 'Summary', 0, 1, 'L')
    pdf.set_font("Arial", size=8)
    pdf.cell(20, 8, 'Team', 1)
    pdf.cell(20, 8, 'xA', 1)
    pdf.cell(20, 8, 'xG', 1)
    pdf.cell(20, 8, 'Territorial possession', 1)

    pdf.ln(8)

    game_merged_df = merged_df[merged_df['label'] == label]
    for index, row in game_merged_df.iterrows():
        pdf.cell(20, 8, row['team_name'], 1)
        pdf.cell(20, 8, f"{row['xA']:.2f}", 1)
        pdf.cell(20, 8, f"{row['xG']:.2f}", 1)
        pdf.cell(20, 8, f"{row['terr_poss']:.2f}", 1)
        pdf.ln(8)

    pdf.output(f"Match reports/Match_Report_{label}.pdf")
    print(f'{label} report created')
# Generate a PDF report for each game involving Horsens
for index, row in horsens_df.iterrows():
    create_pdf_report(row, df_xg_agg, df_xa_agg, merged_df, df_possession_stats)

folder_path = 'C:/Users/SéamusPeareBartholdy/Documents/GitHub/AC-Horsens-First-Team/'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file in files:
    # Check if the file is a PNG file and not 'logo.png'
    if file.endswith(".png") and file != "logo.png":
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file)
        # Remove the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        
# Calculate total combined expected points for Horsens
total_expected_points_combined = total_expected_points_xg.merge(total_expected_points_xa, on='team_name', suffixes=('_xg', '_xa'))
total_expected_points_combined['total_expected_points'] = (total_expected_points_combined['total_expected_points_xg'] + total_expected_points_combined['total_expected_points_xa']) / 2
total_expected_points_combined = total_expected_points_combined[['team_name', 'total_expected_points']]

# Print the total combined expected points for Horsens
print("/nTotal Combined Expected Points for Horsens:")
print(total_expected_points_combined)
