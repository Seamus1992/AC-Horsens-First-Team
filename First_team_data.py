import streamlit as st
import pandas as pd
from mplsoccer import Pitch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(layout='wide')

@st.cache_data
def load_packing_data():
    df_packing = pd.read_csv(r'DNK_1_Division_2023_2024/packing_all DNK_1_Division_2023_2024.csv')
    df_packing['label'] = (df_packing['label'] + ' ' + df_packing['date']).astype(str)
    df_packing = df_packing.rename(columns={'teamName': 'team_name'})
    return df_packing    
@st.cache_data
def load_spacecontrol_data():
    df_spacecontrol = pd.read_csv(r'DNK_1_Division_2023_2024/Space_control_all DNK_1_Division_2023_2024.csv')
    df_spacecontrol['label'] = (df_spacecontrol['label'] + ' ' + df_spacecontrol['date']).astype(str)
    df_spacecontrol = df_spacecontrol.rename(columns={'teamName': 'team_name'})
    return df_spacecontrol   
@st.cache_data
def load_match_stats(columns=None):
    match_stats = pd.read_csv(r'DNK_1_Division_2023_2024/matchstats_all DNK_1_Division_2023_2024.csv')
    match_stats['label'] = (match_stats['label'] + ' ' + match_stats['date'])
    return match_stats
@st.cache_data
def load_possession_data():
    df_possession = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_possession_data.csv')
    df_possession['label'] = (df_possession['label'] + ' ' + df_possession['date']).astype(str)
    #df_possession['team_name'].str.replace(' ', '_')
    return df_possession
@st.cache_data
def load_modstander():
    team_names = ['AaB','B_93','Fredericia','HB_Køge','Helsingør','Hillerød','Hobro','Horsens','Kolding','Næstved','SønderjyskE','Vendsyssel']  # Replace with your list of team names
    Modstander = st.selectbox('Choose opponent',team_names)
    return Modstander
@st.cache_data
def load_possession_stats():
    df_possession_stats = pd.read_csv(r'DNK_1_Division_2023_2024/possession_stats_all DNK_1_Division_2023_2024.csv')
    df_possession_stats['label'] = (df_possession_stats['label'] + ' ' + df_possession_stats['date'])
    return df_possession_stats
@st.cache_data
def load_xg():
    df_xg = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_xg_data.csv')
    df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])
    df_xg['team_name'].str.replace(' ', '_')
    df_xg = df_xg[['playerName','label','team_name','x','y','321','periodId','timeMin','timeSec','9','24','25','26']]
    return df_xg
@st.cache_data
def load_all_xg():
    xg = pd.read_csv(r'DNK_1_Division_2023_2024/xg_all DNK_1_Division_2023_2024.csv')
    xg['label'] = (xg['label'] + ' ' + xg['date'])
    xg['team_name'].str.replace(' ', '_')

    return xg
@st.cache_data
def load_pv():
    df_pv = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_pv_data.csv')
    df_pv['label'] = (df_pv['label'] + ' ' + df_pv['date'])
    df_pv['id'] = df_pv['id'].astype(str)
    df_pv['team_name'].str.replace(' ', '_')
    return df_pv
@st.cache_data
def load_xA():
    df_xA = pd.read_csv(f'DNK_1_Division_2023_2024/xA_all DNK_1_Division_2023_2024.csv')
    df_xA['label'] = (df_xA['label'] + ' ' + df_xA['date']).astype(str)
    return df_xA

def Dashboard():
    df_xg = load_xg()
    df_pv = load_pv()
    df_possession_stats = load_possession_stats()
    df_possession = load_possession_data()
    df_possession['team_name'] = df_possession['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    df_xg['team_name'] = df_xg['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')

    df_matchstats = load_match_stats()
    df_packing = load_packing_data()
    df_xA = load_xA()
    df_spacecontrol = load_spacecontrol_data()
    
    st.title('AC Horsens First Team Dashboard')
    df_possession['date'] = pd.to_datetime(df_possession['date'])
    matches = df_possession['label'].unique()
    matches = matches[::-1]
    match_choice = st.multiselect('Choose a match', matches)
    df_xg = df_xg[df_xg['label'].isin(match_choice)]
    df_pv = df_pv[df_pv['label'].isin(match_choice)]
    df_possession_stats = df_possession_stats[df_possession_stats['label'].isin(match_choice)]
    df_packing = df_packing[df_packing['label'].isin(match_choice)]
    df_matchstats = df_matchstats[df_matchstats['label'].isin(match_choice)]
    df_possession = df_possession[df_possession['label'].isin(match_choice)]
    df_xA = df_xA[df_xA['label'].isin(match_choice)]
    df_spacecontrol = df_spacecontrol[df_spacecontrol['label'].isin(match_choice)]
    df_spacecontrol = df_spacecontrol[df_spacecontrol['Type'] == 'Player']
    df_spacecontrol = df_spacecontrol[['Team','TotalControlArea','CenterControlArea','PenaltyAreaControl','label']]
    df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']] = df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']].astype(float).round(2)
    df_spacecontrol['Team'] = df_spacecontrol['Team'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    
    df_spacecontrol = df_spacecontrol.groupby(['Team', 'label']).sum().reset_index()    
    df_spacecontrol['TotalControlArea_match'] = df_spacecontrol.groupby('label')['TotalControlArea'].transform('sum')
    df_spacecontrol['CenterControlArea_match'] = df_spacecontrol.groupby('label')['CenterControlArea'].transform('sum')
    df_spacecontrol['PenaltyAreaControl_match'] = df_spacecontrol.groupby('label')['PenaltyAreaControl'].transform('sum')

    df_spacecontrol['Total Control Area %'] = df_spacecontrol['TotalControlArea'] / df_spacecontrol['TotalControlArea_match'] * 100
    df_spacecontrol['Center Control Area %'] = df_spacecontrol['CenterControlArea'] / df_spacecontrol['CenterControlArea_match'] * 100
    df_spacecontrol['Penalty Area Control %'] = df_spacecontrol['PenaltyAreaControl'] / df_spacecontrol['PenaltyAreaControl_match'] * 100
    df_spacecontrol = df_spacecontrol[['Team', 'label', 'Total Control Area %', 'Center Control Area %', 'Penalty Area Control %']]
    df_spacecontrol = df_spacecontrol.rename(columns={'Team': 'team_name'})

    xA_map = df_xA[['contestantId','team_name']]
    df_matchstats = df_matchstats.merge(xA_map, on='contestantId', how='inner')
    df_matchstats = df_matchstats.drop_duplicates()
    df_matchstats['team_name'] = df_matchstats['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    df_passes = df_matchstats[['team_name','label','openPlayPass','successfulOpenPlayPass']]

    df_passes = df_passes.groupby(['team_name','label']).sum().reset_index()

    df_xA_summary = df_possession.groupby(['team_name','label'])['318.0'].sum().reset_index()
    df_xA_summary = df_xA_summary.rename(columns={'318.0': 'xA'})

    df_xg_summary = df_xg.groupby(['team_name','label'])['321'].sum().reset_index()
    df_xg_summary = df_xg_summary.rename(columns={'321': 'xG'})
    df_packing_summary = df_packing[['team_name','label','bypassed_opponents','bypassed_defenders']]
    df_packing_summary['team_name'] = df_packing_summary['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')

    df_packing_summary = df_packing_summary.groupby(['team_name','label']).sum().reset_index()
    
    team_summary = df_xg_summary.merge(df_xA_summary, on=['team_name','label'])
    team_summary = team_summary.merge(df_passes, on=['team_name','label'])
    team_summary = team_summary.merge(df_packing_summary, on=['team_name', 'label'])
    team_summary = team_summary.merge(df_spacecontrol, on=['team_name', 'label'])
    team_summary = team_summary.drop(columns='label')
    team_summary = team_summary.groupby('team_name').mean().reset_index()
    team_summary = team_summary.round(2)
    st.dataframe(team_summary.style.format(precision=2), use_container_width=True,hide_index=True)

    
    st.cache_data(experimental_allow_widgets=True)
    st.cache_resource(experimental_allow_widgets=True)
    def xg():
        df_xg = load_xg()
        xg_all = load_all_xg()
        xg_all = xg_all[~(xg_all[['9','24', '25', '26']] == True).any(axis=1)]

        df_xg = df_xg[df_xg['label'].isin(match_choice)]
        df_xg = df_xg[~(df_xg[['9','24', '25', '26']] == True).any(axis=1)]

        xg_period = df_xg[['team_name','321','label']]
        xg_period = xg_period.groupby(['team_name', 'label']).sum().reset_index()
        xg_period['xG_match'] = xg_period.groupby('label')['321'].transform('sum')
        xg_period['xG difference period'] = xg_period['321'] - xg_period['xG_match'] + xg_period['321']
        xg_period = xg_period.groupby('team_name').sum().reset_index()
        xg_period = xg_period[['team_name', 'xG difference period']]
        xg_period = xg_period.sort_values(by=['xG difference period'], ascending=False)
        xg_period = xg_period[xg_period['team_name'] == 'Horsens']
        xg_period['xG difference period'] = xg_period['xG difference period'].round(2)
        
        xg_all = xg_all[['team_name','321','label','date']]
        xg_all = xg_all.groupby(['team_name','label','date']).sum().reset_index()
        xg_all['xG_match'] = xg_all.groupby('label')['321'].transform('sum')
        xg_all['xG difference'] = xg_all['321'] - xg_all['xG_match'] + xg_all['321']
        xg_all = xg_all.sort_values(by=['date'], ascending=True)
        xg_all_table = xg_all.groupby('team_name').sum().reset_index()
        xg_all_table = xg_all_table[['team_name', 'xG difference']]
        xg_all_table = xg_all_table.sort_values(by=['xG difference'], ascending=False)
        xg_all_table['xG difference'] = xg_all_table['xG difference'].round(2)
        xg_all_table['xG difference rank'] = xg_all_table['xG difference'].rank(ascending=False)
        st.dataframe(xg_all_table,hide_index=True)

        xg_all['xG rolling average'] = xg_all.groupby('team_name')['xG difference'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        fig = go.Figure()
        
        for team in xg_all['team_name'].unique():
            team_data = xg_all[xg_all['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Larger line for Horsens
            fig.add_trace(go.Scatter(
                x=team_data['date'], 
                y=team_data['xG rolling average'], 
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))
        
        fig.update_layout(
            title='3-Game Rolling Average of xG Difference Over Time',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average xG Difference',
            template='plotly_white'
        )
        
        st.plotly_chart(fig)

        df_xg['team_name'] = df_xg['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
        df_xg = df_xg.sort_values(by=['team_name','timeMin'])

        df_xg['cumulative_xG'] = df_xg.groupby(['team_name'])['321'].cumsum()

        fig = go.Figure()
        
        for team in df_xg['team_name'].unique():
            team_data = df_xg[df_xg['team_name'] == team]
            fig.add_trace(go.Scatter(
                x=team_data['timeMin'], 
                y=team_data['cumulative_xG'], 
                mode='lines',
                name=team,
            ))
        
        fig.update_layout(
            title='Average Cumulative xG Over Time',
            xaxis_title='Time (Minutes)',
            yaxis_title='Average Cumulative xG',
            template='plotly_white'
        )
        st.dataframe(xg_period, hide_index=True)        
        st.plotly_chart(fig)
    
        df_xg_plot = df_xg[['playerName','team_name','x','y', '321']]
        df_xg_plot = df_xg_plot[df_xg_plot['team_name'] == 'Horsens']
        pitch = Pitch(pitch_type='wyscout',half=True,line_color='white', pitch_color='grass')
        fig, ax = pitch.draw(figsize=(10, 6))
        
        sc = ax.scatter(df_xg_plot['x'], df_xg_plot['y'], s=df_xg_plot['321'] * 100, c='red', edgecolors='black', alpha=0.6)
        
        for i, row in df_xg_plot.iterrows():
            ax.text(row['x'], row['y'], f"{row['playerName']}\n{row['321']:.2f}", fontsize=6, ha='center', va='center')
        
        st.pyplot(fig)
        
    st.cache_data(experimental_allow_widgets=True)
    st.cache_resource(experimental_allow_widgets=True)
    def passes():
        
        df_matchstats = load_match_stats(columns=['contestantId', 'label', 'successfulOpenPlayPass', 'openPlayPass'])
        df_possession = load_possession_data()
        xA_map = df_xA[['contestantId', 'team_name']].drop_duplicates()
        df_matchstats = df_matchstats.merge(xA_map, on='contestantId')
        st.dataframe(df_matchstats, hide_index=True)
        df_possession = df_possession[~(df_possession[['6.0','107.0']] == True).any(axis=1)]
        df_passes_all = df_possession[df_possession['typeId'] == 1]
        df_possession = df_possession[df_possession['label'].isin(match_choice)]
        df_passes_horsens = df_possession[df_possession['team_name'] == 'Horsens']
        df_passes_horsens = df_passes_horsens[(df_passes_horsens['typeId'] == 1) & (df_passes_horsens['outcome'] == 1)]

        mid_third_pass_ends = df_passes_horsens[
            (df_passes_horsens['140.0'].astype(float) >= 33.3) & 
            (df_passes_horsens['140.0'].astype(float) <= 66.3) & 
            (df_passes_horsens['141.0'].astype(float) >= 21.1) & 
            (df_passes_horsens['141.0'].astype(float) <= 78.9) & 
            (df_passes_horsens['y'].astype(float) <= 21.1) | 
            (df_passes_horsens['y'].astype(float) >= 78.9)
        ]
        mid_third_pass_ends = mid_third_pass_ends[['team_name','playerName','eventId', '140.0', '141.0','x', 'y','label','date']]
        # Tæl forekomster af kombinationer af team_name og label
        team_counts = mid_third_pass_ends.groupby(['team_name','label']).size().reset_index(name='count')
        team_counts.columns = ['team_name', 'label', 'count']
        team_counts = team_counts.sort_values(by=['count'], ascending=False)

        # Tæl forekomster af hver playerName
        player_counts = mid_third_pass_ends['playerName'].value_counts().reset_index(name='count')
        player_counts.columns = ['playerName', 'count']
        st.write('Passes from side to halfspace/centerspace')
        st.dataframe(player_counts,hide_index=True)
        st.dataframe(team_counts,hide_index=True)
        
    Data_types = {
        'xG': xg,
        'Passing':passes,
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_data1 = st.selectbox('Choose data type 1', list(Data_types.keys()))
        Data_types[selected_data1]()

    with col2:
        selected_data2 = st.selectbox('Choose data type 2', list(Data_types.keys()))
        Data_types[selected_data2]()

    with col3:
        selected_data3 = st.selectbox('Choose data type 3', list(Data_types.keys()))
        Data_types[selected_data3]()

def League_stats():
    matchstats_df = pd.read_csv(r'DNK_1_Division_2023_2024/matchstats_all DNK_1_Division_2023_2024.csv')
    matchstats_df = matchstats_df.rename(columns={'player_matchName': 'playerName'})
    matchstats_df = matchstats_df.groupby(['contestantId','label', 'date']).sum().reset_index()
    matchstats_df['label'] = np.where(matchstats_df['label'].notnull(), 1, matchstats_df['label'])
    date_format = '%Y-%m-%d'
    matchstats_df['date'] = pd.to_datetime(matchstats_df['date'], format=date_format)
    min_date = matchstats_df['date'].min()
    max_date = matchstats_df['date'].max()

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    date_options = date_range.strftime(date_format)  # Convert dates to the specified format

    default_end_date = date_options[-1]

    default_end_date_dt = pd.to_datetime(default_end_date, format=date_format)
    default_start_date_dt = default_end_date_dt - pd.Timedelta(days=14)  # Subtract 14 days
    default_start_date = default_start_date_dt.strftime(date_format)  # Convert to string

    # Set the default start and end date values for the select_slider
    selected_start_date, selected_end_date = st.select_slider(
        'Choose dates',
        options=date_options,
        value=(default_start_date, default_end_date)
    )

    selected_start_date = pd.to_datetime(selected_start_date, format=date_format)
    selected_end_date = pd.to_datetime(selected_end_date, format=date_format)
    filtered_data = matchstats_df[
        (matchstats_df['date'] >= selected_start_date) & (matchstats_df['date'] <= selected_end_date)
    ]    
    
    xg_df = pd.read_csv(r'DNK_1_Division_2023_2024/xg_all DNK_1_Division_2023_2024.csv')
    xg_df_openplay = xg_df[xg_df['321'] > 0]

    xg_df_openplay = xg_df_openplay.groupby(['contestantId', 'team_name', 'date'])['321'].sum().reset_index()
    xg_df_openplay = xg_df_openplay.rename(columns={'321': 'open play xG'})
    xg_df_openplay['date'] = pd.to_datetime(xg_df_openplay['date'])
        
    matchstats_df = xg_df_openplay.merge(filtered_data)
    matchstats_df = matchstats_df.drop(columns='date')
    matchstats_df = matchstats_df.groupby(['contestantId', 'team_name']).sum().reset_index()
    matchstats_df = matchstats_df.rename(columns={'label': 'matches'})
    matchstats_df['PenAreaEntries per match'] = matchstats_df['penAreaEntries'] / matchstats_df['matches']
    matchstats_df['Open play xG per match'] = matchstats_df['open play xG'] / matchstats_df['matches']
    matchstats_df['Duels per match'] = (matchstats_df['duelLost'] + matchstats_df['duelWon']) /matchstats_df['matches']
    matchstats_df['Duels won %'] = (matchstats_df['duelLost'] + matchstats_df['duelWon']) / matchstats_df['duelWon']
    matchstats_df['Passes per game'] = matchstats_df['openPlayPass'] / matchstats_df['matches']
    matchstats_df['Pass accuracy %'] = matchstats_df['successfulOpenPlayPass'] / matchstats_df['openPlayPass']
    matchstats_df['Back zone pass accuracy %'] = matchstats_df['accurateBackZonePass'] / matchstats_df['totalBackZonePass']
    matchstats_df['Forward zone pass accuracy %'] = matchstats_df['accurateFwdZonePass'] / matchstats_df['totalFwdZonePass']
    matchstats_df['possWonDef3rd %'] = matchstats_df['possWonDef3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['possWonMid3rd %'] = matchstats_df['possWonMid3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['possWonAtt3rd %'] = matchstats_df['possWonAtt3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['Forward pass share %'] = matchstats_df['fwdPass'] / matchstats_df['openPlayPass']
    matchstats_df['Final third entries per match'] = matchstats_df['finalThirdEntries'] / matchstats_df['matches']
    matchstats_df['Final third pass accuracy %'] = matchstats_df['successfulFinalThirdPasses'] / matchstats_df['totalFinalThirdPasses']
    matchstats_df['Open play shot assists share'] = matchstats_df['attAssistOpenplay'] / matchstats_df['totalAttAssist']
    matchstats_df = matchstats_df.drop(columns=['contestantId','playerName','player_playerId','match_id','player_position','player_positionSide','player_subPosition'])
    
    cols_to_rank = matchstats_df.drop(columns=['team_name','matches','minsPlayed']).columns
    ranked_df = matchstats_df.copy()  # Create a copy of the original DataFrame
    for col in cols_to_rank:
        ranked_df[col + '_rank'] = matchstats_df[col].rank(axis=0, ascending=False)
    matchstats_df = ranked_df.merge(matchstats_df)
    matchstats_df = matchstats_df.set_index('team_name')

    st.dataframe(matchstats_df)
    matchstats_df = matchstats_df.reset_index()
    selected_team = st.selectbox('Choose team',matchstats_df['team_name'])  # Replace 'Team A' with the selected team name

    team_df = matchstats_df.loc[matchstats_df['team_name'] == selected_team]

    target_ranks = [1, 2, 3, 10, 11, 12]

    filtered_data_df = pd.DataFrame()

    for col in team_df.columns:
        # Check if the column ends with '_rank' and if any element in the series satisfies the condition
        if col.endswith('_rank') and any(team_df[col].isin(target_ranks)):
            # Extract the corresponding column name without the '_rank' suffix
            original_col = col[:-5]
            # Filter the ranks based on the target ranks
            filtered_ranks = team_df.loc[team_df[col].isin(target_ranks), col]
            # Filter the corresponding values based on the filtered ranks
            filtered_values = team_df.loc[team_df[col].isin(target_ranks), original_col]
            # Add both the filtered ranks and values to the filtered_data_df DataFrame
            filtered_data_df[original_col + '_rank'] = filtered_ranks
            filtered_data_df[original_col + '_value'] = filtered_values
    filtered_data_df = filtered_data_df.T
    filtered_data_df = filtered_data_df.rename(columns={'6': 'value'})
    st.dataframe(filtered_data_df)
    
Data_types = {
    'Dashboard': Dashboard,
    'League stats': League_stats
}

selected_data = st.sidebar.radio('Choose data type',list(Data_types.keys()))
Data_types[selected_data]()
