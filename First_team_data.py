import streamlit as st
import pandas as pd
from mplsoccer import Pitch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(layout='wide')

def load_match_stats():
    match_stats = pd.read_csv(r'DNK_1_Division_2023_2024/matchstats_all DNK_1_Division_2023_2024.csv')
    match_stats['label'] = (match_stats['label'] + ' ' + match_stats['date'])
    return match_stats

def load_possession_data():
    df_possession = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_possession_data.csv')
    df_possession['label'] = (df_possession['label'] + ' ' + df_possession['date']).astype(str)
    df_possession['team_name'].str.replace(' ', '_')
    return df_possession

def load_modstander_possession_data(Modstander):
    df_possession_modstander = pd.read_csv(f'1. Division/{Modstander}/{Modstander}_possession_data.csv')
    df_possession_modstander['label'] = (df_possession_modstander['label'] + ' ' + df_possession_modstander['date']).astype(str)
    df_possession_modstander['team_name'].str.replace(' ', '_')

    return df_possession_modstander

def load_modstander():
    team_names = ['AaB','B_93','Fredericia','HB_Køge','Helsingør','Hillerød','Hobro','Horsens','Kolding','Næstved','SønderjyskE','Vendsyssel']  # Replace with your list of team names
    Modstander = st.selectbox('Choose opponent',team_names)
    return Modstander

def load_possession_stats():
    df_possession_stats = pd.read_csv(r'DNK_1_Division_2023_2024/possession_stats_all DNK_1_Division_2023_2024.csv')
    df_possession_stats['label'] = (df_possession_stats['label'] + ' ' + df_possession_stats['date'])
    return df_possession_stats

def load_xg():
    df_xg = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_xg_data.csv')
    df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])
    df_xg['team_name'].str.replace(' ', '_')
    return df_xg

def load_all_xg():
    xg = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_xg_data.csv')
    xg['label'] = (xg['label'] + ' ' + xg['date'])
    xg['team_name'].str.replace(' ', '_')
    return xg

def load_xg_opponent(Modstander):
    df_xg_opponent = pd.read_csv(f'DNK_1_Division_2023_2024/{Modstander}/{Modstander}_xg_data.csv')
    df_xg_opponent['label'] = (df_xg_opponent['label'] + ' ' + df_xg_opponent['date'])
    df_xg_opponent['team_name'].str.replace(' ', '_')
    return df_xg_opponent

def load_pv():
    df_pv = pd.read_csv(r'DNK_1_Division_2023_2024/Horsens/Horsens_pv_data.csv')
    df_pv['label'] = (df_pv['label'] + ' ' + df_pv['date'])
    df_pv['id'] = df_pv['id'].astype(str)
    df_pv['team_name'].str.replace(' ', '_')
    return df_pv

def load_pv_opponent(Modstander):
    df_pv_opponent = pd.read_csv(f'DNK_1_Division_2023_2024/{Modstander}/{Modstander}_pv_data.csv')
    df_pv_opponent['label'] = (df_pv_opponent['label'] + ' ' + df_pv_opponent['date'])
    df_pv_opponent['id'] = df_pv_opponent['id'].astype(str)
    df_pv_opponent['team_name'].str.replace(' ', '_')
    return df_pv_opponent

@st.cache_data(experimental_allow_widgets=True)
@st.cache_resource(experimental_allow_widgets=True)
def Match_evaluation ():
    team_name = 'Horsens'    
    df_pv = load_pv()
    df_xg = load_xg()
    df_possession_stats = load_possession_stats()
    df_possession = load_possession_data()

    Hold = df_pv['team_name'].unique()
    Hold = [team.replace(' ', '_') for team in Hold]
    Hold = sorted(Hold)
    Modstander = team_name
    Kampe = df_pv[df_pv['team_name'].astype(str) == Modstander]
    Kampe = Kampe.sort_values(by='date',ascending = False)
    Kampe_labels = Kampe['label'].unique()

    Kampvalg = st.selectbox('Vælg kampe',Kampe_labels)


    df_pv = df_pv[df_pv['label'] == Kampvalg]
    df_xg = df_xg[df_xg['label'] == Kampvalg]
    df_xg = df_xg[(df_xg[['9', '24', '25', '26']] != True).all(axis=1)]
    st.dataframe(df_xg)
    df_possession_stats = df_possession_stats[df_possession_stats['label'] == Kampvalg]
    df_possession = df_possession[df_possession['label'] == Kampvalg]

    
    df_possession_stats = df_possession_stats[df_possession_stats['type'] == 'territorialThird']
    df_possession_stats['home'] = df_possession_stats['home'].astype(float)
    df_possession_stats['away'] = df_possession_stats['away'].astype(float)
    df_possession_home = df_possession_stats['home'].mean()
    df_possession_away = df_possession_stats['away'].mean()
    df_possession_stats_summary = pd.DataFrame({'home': [df_possession_home], 'away': [df_possession_away]})
    first_home_team = df_possession_stats['home_team'].iloc[0]
    first_away_team = df_possession_stats['away_team'].iloc[0]
    df_possession_stats = df_possession_stats.rename(columns={'home': first_home_team, 'away': first_away_team})
    df_possession_stats_summary = df_possession_stats_summary.rename(columns={'home': first_home_team, 'away': first_away_team})
    df_possession_stats = df_possession_stats.drop_duplicates()
    df_possession_stats = df_possession_stats[df_possession_stats['interval_type'] == 5]
    df_possession_stats_summary = df_possession_stats_summary.transpose().reset_index()
    df_possession_stats_summary = df_possession_stats_summary.rename(columns={'index':'team_name',0:'terr_poss'})
    
    
    df_possession['id'] = df_possession['id'].astype(str)
    df_possession = df_possession.astype(str)
    df_pv = df_pv[['team_name','playerName','id','possessionValue.pvValue','possessionValue.pvAdded']].astype(str)

    df_possession_pv = pd.merge(df_possession,df_pv,how='outer')
    df_possession_pv['PvTotal'] = df_possession_pv['possessionValue.pvValue'].astype(float) + df_possession_pv['possessionValue.pvAdded'].astype(float)
    df_possession_pv_hold = df_possession_pv[df_possession_pv['label'] == Kampvalg]
    df_possession_pv_hold = df_possession_pv_hold.drop_duplicates('id')
    df_pv_agg = df_possession_pv_hold[['team_name','label','PvTotal','timeMin','timeSec']]
    df_pv_agg.loc[:, 'timeMin'] = df_pv_agg['timeMin'].astype(int)
    df_pv_agg.loc[:, 'timeSec'] = df_pv_agg['timeSec'].astype(int)
    df_pv_agg = df_pv_agg.sort_values(by=['timeMin', 'timeSec'])
    df_pv_agg = df_pv_agg[df_pv_agg['PvTotal'].astype(float) > 0]
    df_pv_agg['culmulativpv'] = df_pv_agg.groupby(['team_name','label'])['PvTotal'].cumsum()
    df_possession_pv_hold = df_possession_pv_hold.groupby(['team_name','label'])['PvTotal'].sum().reset_index()

    df_xg_hold = df_xg[df_xg['label'] == Kampvalg]
    df_xg_hold = df_xg_hold.rename(columns={'321': 'Open play xG'})
    df_xg_agg = df_xg_hold[['team_name','Open play xG','periodId','timeMin','timeSec']]
    df_xg_agg.loc[:,'timeMin'] = df_xg_agg['timeMin'].astype(int)
    df_xg_agg.loc[:,'timeSec'] = df_xg_agg['timeSec'].astype(int)
    df_xg_agg = df_xg_agg.sort_values(by=['timeMin', 'timeSec'])

    df_xg_agg = df_xg_agg[df_xg_agg['Open play xG'].astype(float) > 0]
    df_xg_agg['culmulativxg'] = df_xg_agg.groupby('team_name')['Open play xG'].cumsum()
    df_xg_hold = df_xg_hold.groupby(['team_name','label'])['Open play xG'].sum().reset_index()
    df_holdsummary = df_xg_hold.merge(df_possession_pv_hold)
    df_xa = df_possession[df_possession['label'] == Kampvalg]
    df_xa['318.0'] = df_xa['318.0'].astype(float)
    df_xa_agg = df_xa[['team_name','318.0','timeMin','timeSec']]
    df_xa_agg = df_xa_agg.rename(columns={'318.0': 'xA'})
    df_xa_agg.loc[:,'timeMin'] = df_xa_agg['timeMin'].astype(int)
    df_xa_agg.loc[:,'timeSec'] = df_xa_agg['timeSec'].astype(int)
    df_xa_agg = df_xa_agg.sort_values(by=['timeMin', 'timeSec'])
    df_xa_agg = df_xa_agg[df_xa_agg['xA'].astype(float) > 0]
    df_xa_agg['culmulativxa'] = df_xa_agg.groupby('team_name')['xA'].cumsum()

    df_xa_hold = df_xa.groupby(['team_name','label'])['318.0'].sum().reset_index()
    df_xa_hold = df_xa_hold.rename(columns={'318.0': 'xA'})
    df_holdsummary = df_xa_hold.merge(df_holdsummary)
    df_holdsummary = df_possession_stats_summary.merge(df_holdsummary)
    df_holdsummary = df_holdsummary[['team_name','label','xA','Open play xG','PvTotal','terr_poss']]
    st.dataframe(df_holdsummary,hide_index=True)
    col1,col2,col3,col4 = st.columns(4)

    cols_to_average = df_possession_stats.columns[-2:]
    time_column = 'interval'

    # Calculate sliding average
    sliding_average = df_possession_stats[cols_to_average].rolling(window=2).mean()

    # Plotting
    fig, ax = plt.subplots()
    for col in sliding_average.columns:
        ax.plot(df_possession_stats[time_column], sliding_average[col], label=f'{col} Sliding Average')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Territorial possession')
    ax.set_title('Sliding average territorial possession')

    # Add legend
    ax.legend()

    # Display the plot using Streamlit
    with col1:
        st.pyplot(fig)

    fig, ax = plt.subplots()

    # Iterate over each team
    for team, data in df_pv_agg.groupby('team_name'):
        ax.plot(data['timeMin'], data['culmulativpv'], label=team)

    # Set labels and title
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative pv')
    ax.set_title('Cumulative pv over time')

    # Add legend
    ax.legend()

    with col2:
        st.pyplot(fig)

    fig, ax = plt.subplots()

    # Iterate over each team
    for team, data in df_xg_agg.groupby('team_name'):
        ax.plot(data['timeMin'], data['culmulativxg'], label=team)

    # Set labels and title
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative xg')
    ax.set_title('Cumulative xg over time')

    # Add legend
    ax.legend()

    # Display the plot
    with col3:
        st.pyplot(fig)

    fig, ax = plt.subplots()

    # Iterate over each team
    for team, data in df_xa_agg.groupby('team_name'):
        ax.plot(data['timeMin'], data['culmulativxa'], label=team)

    # Set labels and title
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Cumulative xA')
    ax.set_title('Cumulative xA over time')

    # Add legend
    ax.legend()

    with col4:
        st.pyplot(fig)

    df_possession_pv['possessionValue.pvValue'] = df_possession_pv['possessionValue.pvValue'].astype(float)
    df_possession_pv = df_possession_pv[df_possession_pv['label'] == Kampvalg]
    df_possession_pv = df_possession_pv[df_possession_pv['team_name'] == Modstander]
    df_xa_player = df_possession_pv[['playerName','318.0']]
    df_xa_player.loc[:, '318.0'] = df_xa_player['318.0'].astype(float)
    df_xa_player = df_xa_player[df_xa_player['playerName'] != 'nan']
    df_xa_player = df_xa_player.groupby('playerName')['318.0'].sum()

    df_possession_pv = df_possession_pv[df_possession_pv['playerName'] != 'nan']
    player_pv_df = df_possession_pv.groupby('playerName')['PvTotal'].sum()
    players = df_possession_pv['playerName'].astype(str).drop_duplicates()
    pitch = Pitch(pitch_type='opta', line_color='white', pitch_color='grass',pad_top=20)
    fig, axs = pitch.grid(ncols=4, nrows=4, grid_height=0.85, title_height=0.00, axis=False, title_space=0.04, endnote_space=0.01)
    plt.figure()

    for name, ax in zip(players, axs['pitch'].flat[:len(players)]):
        player_df = df_possession_pv.loc[df_possession_pv["playerName"] == name]
        PV_score = player_pv_df[name]  # Fetch PV score for the player
        xa_score = df_xa_player[name]
        ax.text(50,110,f"{name} ({PV_score:.3f} PV) ({xa_score:.5f} xA)",ha='center',va='center', fontsize=8, color='white')

        for i in player_df.index:
            x = player_df['x'].astype(float)[i]
            y = player_df['y'].astype(float)[i]
            dx_pass = player_df['140.0'].astype(float)[i] - player_df['x'].astype(float)[i]
            dy_pass = player_df['141.0'].astype(float)[i] - player_df['y'].astype(float)[i]
            arrow_color = 'red' if player_df['outcome'].astype(int)[i] == 0 else '#0dff00'
            ax.arrow(x, y, dx_pass, dy_pass, color=arrow_color, length_includes_head=True, head_width=0.5, head_length=0.5)
            pitch.scatter(player_df['x'].astype(float)[i], player_df['y'].astype(float)[i], color=arrow_color,s=20, ax=ax)

    st.title('Passes, pv and xA')
    st.pyplot(fig)
    plt.close(fig)

    df_keypass = df_possession[df_possession['team_name'] == Modstander]
    df_keypass = df_keypass[df_keypass['label'] == Kampvalg]
    df_keypass = df_keypass[df_keypass['q_qualifierId'] == '210.0']
    df_keypass = df_keypass.drop_duplicates('id')
    df_keypass_spiller = df_keypass['playerName'].value_counts()
    df_keypass_spiller = df_keypass_spiller.sort_values(ascending=False)

    df_pv_modstander = df_pv[df_pv['team_name'] == Modstander]
    df_pv_modstander.loc[:,'possessionValue.pvAdded'] = df_pv_modstander['possessionValue.pvAdded'].astype(float)
    df_pv_spiller = df_pv_modstander.groupby('playerName')['possessionValue.pvAdded'].sum()
    df_pv_spiller = df_pv_spiller.sort_values(ascending=False)

    df_xg_modstander = df_xg[df_xg['team_name'].astype(str) == Modstander]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['label']==Kampvalg]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_value'].astype(float) > 0]
    df_xg_modstander['q_value'] = df_xg_modstander['q_value'].astype(float)
    df_xg_spiller = df_xg_modstander.groupby('playerName')['q_value'].sum()
    df_xg_spiller = df_xg_spiller.sort_values(ascending=False)

    df_assist = df_possession.copy()
    df_assist = df_assist[df_assist['team_name'].astype(str) == Modstander]
    try:
        df_assist['assist'] = df_assist['assist'].astype(float)
        df_assist = df_assist[df_assist['assist'] == 1]
        df_assist = df_assist.drop_duplicates('id')
        df_assist_spiller = df_assist.groupby('playerName')['assist'].sum()
        df_assist_spiller = df_assist_spiller.sort_values(ascending=False)
    except KeyError:
        df_assist = pd.DataFrame()

    df_xg_plot = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot = df_xg_plot[df_xg_plot['q_value'].astype(float) > 0.0]

    col1,col2 = st.columns(2)
    x = df_xg_plot['x'].astype(float)
    y = df_xg_plot['y'].astype(float)
    shot_xg = df_xg_plot['q_value'].astype(float)
    player_names = df_xg_plot['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)
    for j, (player_names, x_val, y_val, xg_val) in enumerate(zip(player_names, x, y, shot_xg)):
        ax.annotate(f'{player_names}\n{xg_val:.2f}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    df_xg_plot = df_xg_plot.set_index('playerName')

    with col1:
        st.write('Xg plot '+ Modstander)
        st.pyplot(fig)

    df_xg_plot_store_chancer = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot_store_chancer = df_xg_plot_store_chancer[df_xg_plot_store_chancer['q_value'].astype(float) > 0.2]


    x = df_xg_plot_store_chancer['x'].astype(float)
    y = df_xg_plot_store_chancer['y'].astype(float)
    shot_xg = df_xg_plot_store_chancer['q_value'].astype(float)
    player_names = df_xg_plot_store_chancer['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)
    with col2:
        st.write('Xg plot big chances ' + Modstander)
        st.pyplot(fig)

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.write('Shot assist per player')
        st.dataframe(df_keypass_spiller)
    with col2:
        st.write('Xg per player')
        st.dataframe(df_xg_spiller)
    with col3:
        st.write('Pvadded per player')
        st.dataframe(df_pv_spiller)
    with col4:
        st.write('Assists per player')
        try:
            st.dataframe(df_assist_spiller)
        except UnboundLocalError:
            st.write('No assists in the game') 

    col1,col2 = st.columns(2)
    player_names = df_keypass['playerName'].astype(str)

    x = df_keypass['x'].astype(float)
    y = df_keypass['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    df_keypass = df_keypass.set_index('playerName')

    with col1:
        st.write('Shot assists')
        st.pyplot(fig)

    player_names = df_assist['playerName'].astype(str)

    x = df_assist['x'].astype(float)
    y = df_assist['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    df_assist = df_assist.set_index('playerName')

    with col2:
        st.write('Assists')
        st.pyplot(fig)
        plt.close(fig)

    df_possession_modstander = df_possession[df_possession['team_name'] == Modstander]
    df_possession_modstander = df_possession_modstander[df_possession_modstander['label'] == Kampvalg]
    #sorterer for standardsituationer
    df_possession_modstander['q_qualifierId'] = df_possession_modstander['q_qualifierId'].astype(float)
    filtered_ids = df_possession_modstander[df_possession_modstander['q_qualifierId'].isin([22,23])]['id']
    # Filter out all rows with the filtered 'id' values
    filtered_data = df_possession_modstander[df_possession_modstander['id'].isin(filtered_ids)]
    #erobringer til store chancer
    df_store_chancer = filtered_data[(filtered_data['q_qualifierId'] == 321)]
    df_store_chancer = df_store_chancer[df_store_chancer['q_value'].astype(float) > 0.01]

    store_chancer_sequencer = df_store_chancer[['label','sequenceId']]
    store_chancer_sequencer = store_chancer_sequencer.merge(df_possession_modstander)
    store_chancer_sequencer = store_chancer_sequencer.drop_duplicates(subset='sequenceId', keep='first')
    player_names = store_chancer_sequencer['playerName'].astype(str)

    x = store_chancer_sequencer['x'].astype(float)
    y = store_chancer_sequencer['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    store_chancer_sequencer = store_chancer_sequencer.set_index('playerName')

    col1,col2 = st.columns([1,1])
    with col1:
        st.write('Ball recoveries/interceptions that lead to a chance (0,01 xg)')
        st.pyplot(fig)
        plt.close(fig)

    store_chancer_sequencer_spillere = store_chancer_sequencer.value_counts('playerName').reset_index()
    store_chancer_sequencer_spillere.columns = ['playerName', 'Number']

    with col2:
        st.write('All ball recoveries/interceptions')
        interceptions_df = df_possession[(df_possession['typeId'].astype(int) == 8) | (df_possession['typeId'].astype(int) == 49)]
        interceptions_df = interceptions_df[interceptions_df['label'] == Kampvalg]
        interceptions_df = interceptions_df[interceptions_df['team_name'] == Modstander]

        x = interceptions_df['x'].astype(float)
        y = interceptions_df['y'].astype(float)

        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
        fig, ax = pitch.draw()
        sc = pitch.scatter(x, y, ax=ax)
        interceptions_df = interceptions_df.set_index('playerName')
        st.pyplot(fig)    
        plt.close(fig)

    st.title('Against ' + Modstander)
    #Modstanders modstandere

    df_keypass = df_possession[df_possession['team_name'] != Modstander]
    df_keypass = df_keypass[df_keypass['label']== Kampvalg]
    df_keypass = df_keypass[df_keypass['q_qualifierId'] == '210.0']
    df_keypass = df_keypass.drop_duplicates('id')
    df_keypass_spiller = df_keypass['playerName'].value_counts()
    df_keypass_spiller = df_keypass_spiller.sort_values(ascending=False)

    df_xg_modstander = df_xg[df_xg['label'].str.contains(Modstander)]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['label'] == Kampvalg]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['team_name'] != Modstander]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_value'].astype(float) > 0]
    df_xg_modstander['q_value'] = df_xg_modstander['q_value'].astype(float)
    df_xg_spiller = df_xg_modstander.groupby('playerName')['q_value'].sum()
    df_xg_spiller = df_xg_spiller.sort_values(ascending=False)

    df_assist = df_possession.copy()
    df_assist = df_assist[df_assist['label'].str.contains(Modstander)]
    df_assist = df_assist[df_assist['label'] == Kampvalg]
    df_assist = df_assist[df_assist['team_name'] != Modstander]
    df_assist['assist'] = df_assist['assist'].astype(float)
    df_assist = df_assist[df_assist['assist'] > 0]
    df_assist_spiller = df_assist.groupby('playerName')['assist'].sum()
    df_assist_spiller = df_assist_spiller.sort_values(ascending=False)

    df_possession = df_possession[df_possession['label'] == Kampvalg]
    df_possession_modstander = df_possession[df_possession['team_name'] != Modstander]
    df_possession_modstander = df_possession_modstander[df_possession_modstander['label'] == Kampvalg]
    df_possession_modstander_xA = df_possession_modstander[df_possession_modstander['q_qualifierId'].astype(float) == 318]
    df_possession_modstander_xA = df_possession_modstander_xA[df_possession_modstander_xA['q_value'].astype(float) > 0.05]

    df_xg_plot = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot = df_xg_plot[df_xg_plot['q_value'].astype(float) > 0.0]

    col1,col2 = st.columns(2)
    x = df_xg_plot['x'].astype(float)
    y = df_xg_plot['y'].astype(float)
    shot_xg = df_xg_plot['q_value'].astype(float)
    player_names = df_xg_plot['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)

    with col1:
        st.write('Xg plot against '+ Modstander)
        st.pyplot(fig)

    df_xg_plot_store_chancer = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot_store_chancer = df_xg_plot_store_chancer[df_xg_plot_store_chancer['q_value'].astype(float) > 0.2]

    x = df_xg_plot_store_chancer['x'].astype(float)
    y = df_xg_plot_store_chancer['y'].astype(float)
    shot_xg = df_xg_plot_store_chancer['q_value'].astype(float)
    player_names = df_xg_plot_store_chancer['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)
    with col2:
        st.write('Xg plot big chances against ' + Modstander)
        st.pyplot(fig)
    
    col1,col2 = st.columns(2)

    x = df_keypass['x'].astype(float)
    y = df_keypass['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    with col1:
        st.write('Shot assists against ' + Modstander)
        st.pyplot(fig)

    x = df_assist['x'].astype(float)
    y = df_assist['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    with col2:
        st.write('Assists against ' + Modstander)
        st.pyplot(fig)
        
    #sorterer for standardsituationer
    df_possession['q_qualifierId'] = df_possession['q_qualifierId'].astype(float)
    filtered_ids = df_possession[df_possession['q_qualifierId'].isin([22,23])]['id']
    # Filter out all rows with the filtered 'id' values
    filtered_data = df_possession[df_possession['id'].isin(filtered_ids)]
    #erobringer til store chancer
    df_store_chancer = filtered_data[(filtered_data['q_qualifierId'] == 321)]
    df_store_chancer = df_store_chancer[df_store_chancer['q_value'].astype(float) > 0.01]
    store_chancer_sequencer = df_store_chancer[['label','sequenceId']]
    store_chancer_sequencer = store_chancer_sequencer.merge(df_possession)
    store_chancer_sequencer = store_chancer_sequencer.drop_duplicates(subset='sequenceId', keep='first')
    x = store_chancer_sequencer['x'].astype(float)
    y = store_chancer_sequencer['y'].astype(float)
    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    col1,col2 = st.columns(2)
    with col1:
        st.write('Interceptions/recoveries that lead to a chance against' + Modstander + ' (0,01 xg)')
        st.pyplot(fig)
    store_chancer_sequencer_spillere = store_chancer_sequencer.value_counts('playerName')

    with col2:
        st.write('All interceptions/recoveries against ' + Modstander)
        interceptions_df = df_possession[(df_possession['typeId'].astype(int) == 8) | (df_possession['typeId'].astype(int) == 49)]
        interceptions_df = interceptions_df[interceptions_df['label'] == Kampvalg]
        interceptions_df = interceptions_df[interceptions_df['team_name'] != Modstander]

        x = interceptions_df['x'].astype(float)
        y = interceptions_df['y'].astype(float)

        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
        fig, ax = pitch.draw()
        sc = pitch.scatter(x, y, ax=ax)
        interceptions_df = interceptions_df.set_index('playerName')
        st.pyplot(fig)        

st.cache_data(experimental_allow_widgets=True)
st.cache_resource(experimental_allow_widgets=True)
def Team_development ():
    xg = load_all_xg()
    match_stats = load_match_stats()

    match_stats['label'] = match_stats['label'] + ' ' + match_stats['date']
    xg['label'] = xg['label'] + ' ' + xg['date']
    xg['xG'] = xg['q_value'].astype(float)
    xg = xg.groupby(['label', 'team_name']).sum().reset_index()
    regression = match_stats.groupby(['label', 'team_name'])['finalThirdEntries'].sum().reset_index()
    regression = regression.merge(xg)

    regression = regression.groupby(['label','team_name']).sum().reset_index()
    fig = px.scatter(regression, x='finalThirdEntries', y='xG', title='Scatterplot of finalThirdEntries vs. expected goals')

    # Fit a linear trendline using numpy polyfit
    m, b = np.polyfit(regression['finalThirdEntries'], regression['xG'], 1)
    fig.add_traces(go.Scatter(x=regression['finalThirdEntries'], y=m * regression['finalThirdEntries'] + b, 
                            mode='lines', name='Trendline'))

    # Update layout
    fig.update_layout(xaxis_title='Final Third Entries', yaxis_title='Expected goals')

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    match_stats = match_stats[match_stats['team_name'] == 'Horsens']
    match_stats = match_stats[['date','label','finalThirdEntries','possWonAtt3rd','totalFwdZonePass','fwdPass','duelWon','possWonMid3rd']]
    match_stats['Interception opp half'] = match_stats['possWonMid3rd'] + match_stats['possWonAtt3rd']
    match_stats = match_stats[['label','finalThirdEntries','Interception opp half','fwdPass','duelWon','totalFwdZonePass']]
    match_stats = match_stats.groupby('label').sum().reset_index()

    # Sort by date
    match_stats['date'] = pd.to_datetime(match_stats['label'].str.split().str[-1])
    match_stats = match_stats.sort_values('date')

    numeric_columns = ['finalThirdEntries','Interception opp half','fwdPass','duelWon','totalFwdZonePass']
    match_stats_rolling = match_stats[numeric_columns].rolling(window=5).mean()

    # Reassign 'label' column to match_stats_rolling DataFrame
    match_stats_rolling['label'] = match_stats['label']
    match_stats_rolling = match_stats_rolling[['label','finalThirdEntries','Interception opp half','fwdPass','duelWon']]
    # Create Plotly figure
    fig = go.Figure()

    # Adding traces for each statistic
    for column in match_stats_rolling.columns[1:]:
        fig.add_trace(go.Scatter(x=match_stats_rolling['label'], y=match_stats_rolling[column], mode='lines', name=column))

    # Adding layout
    fig.update_layout(title='5-Game Rolling Average Match Stats of Horsens',
                    xaxis_title='Match',
                    yaxis_title='Stat',
                    legend=dict(x=0, y=1, traceorder='normal'))

    # Display Plotly chart
    st.plotly_chart(fig,use_container_width=True)


    possession_stats = pd.read_csv(r'1. Division/possession_stats_all 1. Division.csv')

    possession_stats = possession_stats[(possession_stats['away_team'] == 'Horsens') | (possession_stats['home_team'] == 'Horsens')]
    possession_stats = possession_stats[possession_stats['type'] == 'territorialThird']

    # Combine 'label' and 'date' to create a unique label for each match
    possession_stats['label'] = possession_stats['label'] + ' ' + possession_stats['date']

    # Select relevant columns
    possession_stats = possession_stats[['label', 'home_team', 'away_team', 'overall.away', 'overall.home', 'overall.middle']]

    # Define function to assign values to 'Horsens' and 'Opponent' columns
    def assign_values(row):
        if row['home_team'] == 'Horsens':
            return row['overall.home'], row['overall.away']
        elif row['away_team'] == 'Horsens':
            return row['overall.away'], row['overall.home']
        else:
            return None, None

    possession_stats['Horsens'], possession_stats['Opponent'] = zip(*possession_stats.apply(assign_values, axis=1))

    # Drop duplicates to keep only the first occurrence of each match
    possession_stats = possession_stats.drop_duplicates(keep='first')

    # Sort by date
    possession_stats['date'] = pd.to_datetime(possession_stats['label'].str.split().str[-1])
    possession_stats = possession_stats.sort_values('date')

    numeric_columns = ['overall.away', 'overall.home', 'overall.middle', 'Horsens', 'Opponent']
    possession_stats_rolling = possession_stats[numeric_columns].rolling(window=5).mean()

    # Reassign 'label' column to possession_stats_rolling DataFrame
    possession_stats_rolling['label'] = possession_stats['label']

    # Create Plotly figure
    fig = go.Figure()

    # Adding traces for 'Horsens' and 'Opponent' with 5-game rolling averages
    fig.add_trace(go.Scatter(x=possession_stats_rolling['label'], y=possession_stats_rolling['Horsens'], mode='lines', name='Horsens (5-Game Rolling Avg)'))
    fig.add_trace(go.Scatter(x=possession_stats_rolling['label'], y=possession_stats_rolling['Opponent'], mode='lines', name='Opponent (5-Game Rolling Avg)'))

    # Adding layout
    fig.update_layout(title='5-Game Rolling Average Territorial possession of Horsens vs. Opponent',
                    xaxis_title='Match',
                    yaxis_title='Territorial Possession',
                    legend=dict(x=0, y=1, traceorder='normal'))

    # Display Plotly chart using Streamlit
    st.plotly_chart(fig,use_container_width=True)

st.cache_data(experimental_allow_widgets=True)
st.cache_resource(experimental_allow_widgets=True)
def Opposition_analysis ():
    import streamlit as st
    import pandas as pd
    from mplsoccer import Pitch
    import numpy as np

    col1,col2 = st.columns(2)
    with col1:
        selected_opponent = load_modstander()
    
    df_pv = load_pv_opponent(selected_opponent)
    df_xg = load_xg_opponent(selected_opponent)
    df_possession_modstander = load_modstander_possession_data(selected_opponent)
    
    Hold = df_pv['team_name'].unique()
    Hold = sorted(Hold)
    Kampe = df_pv[df_pv['team_name'].astype(str) == selected_opponent]
    Kampe = Kampe.sort_values(by='date',ascending = False)
    Kampe_labels = Kampe['label'].unique()
    Kampe_labels = Kampe_labels.astype(str)

    with col2:
        Kampvalg = st.multiselect('Choose matches (last 5 per default)', Kampe_labels, default=Kampe_labels[:5])

    df_possession_modstander = df_possession_modstander[df_possession_modstander['team_name'] == selected_opponent]
    df_possession_modstander = df_possession_modstander[df_possession_modstander['label'].isin(Kampvalg)]

    df_keypass = df_possession_modstander[df_possession_modstander['team_name'] == selected_opponent]
    df_keypass = df_keypass[df_keypass['label'].isin(Kampvalg)]
    df_keypass = df_keypass[df_keypass['q_qualifierId'] == 210]
    df_keypass = df_keypass.drop_duplicates('id')
    df_keypass_spiller = df_keypass['playerName'].value_counts()
    df_keypass_spiller = df_keypass_spiller.sort_values(ascending=False)

    df_pv_modstander = df_pv[df_pv['team_name'] == selected_opponent]
    df_pv_modstander = df_pv_modstander[df_pv_modstander['label'].isin(Kampvalg)]
    df_pv_modstander['possessionValue.pvAdded'] = df_pv_modstander['possessionValue.pvAdded'].astype(float)
    df_pv_spiller = df_pv_modstander.groupby('playerName')['possessionValue.pvAdded'].sum()
    df_pv_spiller = df_pv_spiller.sort_values(ascending=False)

    df_xg_modstander = df_xg[df_xg['team_name'] == selected_opponent]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['label'].isin(Kampvalg)]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321.0]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_value'].astype(float) > 0]
    df_xg_modstander['q_value'] = df_xg_modstander['q_value'].astype(float)
    df_xg_spiller = df_xg_modstander.groupby('playerName')['q_value'].sum()
    df_xg_spiller = df_xg_spiller.sort_values(ascending=False)

    df_assist = df_possession_modstander.copy()
    df_assist = df_assist[df_assist['team_name'] == selected_opponent]
    df_assist['assist'] = df_assist['assist'].astype(float)
    df_assist = df_assist[df_assist['assist'] == 1]
    df_assist = df_assist.drop_duplicates('id')
    df_assist_spiller = df_assist.groupby('playerName')['assist'].sum()
    df_assist_spiller = df_assist_spiller.sort_values(ascending=False)

    df_possession_modstander = df_possession_modstander[df_possession_modstander['team_name'] == selected_opponent]
    df_possession_modstander = df_possession_modstander[df_possession_modstander['label'].isin(Kampvalg)]

    df_xg_plot = df_xg_modstander
    df_xg_plot = df_xg_plot[df_xg_plot['q_value'].astype(float) > 0.0]

    col1,col2 = st.columns(2)
    x = df_xg_plot['x'].astype(float)
    y = df_xg_plot['y'].astype(float)
    shot_xg = df_xg_plot['q_value'].astype(float)
    player_names = df_xg_plot['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)

    with col1:
        st.write('Xg plot '+ selected_opponent)
        st.pyplot(fig)

    df_xg_plot_store_chancer = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot_store_chancer = df_xg_plot_store_chancer[df_xg_plot_store_chancer['q_value'].astype(float) > 0.2]

    x = df_xg_plot_store_chancer['x'].astype(float)
    y = df_xg_plot_store_chancer['y'].astype(float)
    shot_xg = df_xg_plot_store_chancer['q_value'].astype(float)
    player_names = df_xg_plot_store_chancer['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)
    with col2:
        st.write('Xg plot store chancer ' + selected_opponent)
        st.pyplot(fig)

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.write('Shot assist per spiller')
        st.dataframe(df_keypass_spiller)
    with col2:
        st.write('Xg per spiller')
        st.dataframe(df_xg_spiller)
    with col3:
        st.write('Pvadded per spiller')
        st.dataframe(df_pv_spiller)
    with col4:
        st.write('Assists per spiller')
        st.dataframe(df_assist_spiller)

    col1,col2 = st.columns(2)
    player_names = df_keypass['playerName'].astype(str)

    x = df_keypass['x'].astype(float)
    y = df_keypass['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    df_keypass = df_keypass.set_index('playerName')

    with col1:
        st.write('Shot assists')
        st.pyplot(fig)
    
    player_names = df_assist['playerName'].astype(str)
    x = df_assist['x'].astype(float)
    y = df_assist['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    df_assist = df_assist.set_index('playerName')

    with col2:
        st.write('Assists')
        st.pyplot(fig)
        
    #sorterer for standardsituationer
    df_possession_modstander['q_qualifierId'] = df_possession_modstander['q_qualifierId'].astype(float)
    filtered_ids = df_possession_modstander[df_possession_modstander['q_qualifierId'].isin([22,23])]['id']
    # Filter out all rows with the filtered 'id' values
    filtered_data = df_possession_modstander[df_possession_modstander['id'].isin(filtered_ids)]
    #erobringer til store chancer
    df_store_chancer = filtered_data[(filtered_data['q_qualifierId'] == 321)]
    df_store_chancer = df_store_chancer[df_store_chancer['q_value'].astype(float) > 0.01]
    store_chancer_sequencer = df_store_chancer[['label','sequenceId']]
    store_chancer_sequencer = store_chancer_sequencer.merge(df_possession_modstander)
    store_chancer_sequencer = store_chancer_sequencer.drop_duplicates(subset='sequenceId', keep='first')
    
    player_names = store_chancer_sequencer['playerName'].astype(str)
    x = store_chancer_sequencer['x'].astype(float)
    y = store_chancer_sequencer['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    for j, (player_names, x_val, y_val) in enumerate(zip(player_names, x, y)):
        ax.annotate(f'{player_names}', (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset pixels', fontsize=8, color='black')    
    store_chancer_sequencer = store_chancer_sequencer.set_index('playerName')

    col1,col2 = st.columns([3,1])
    with col1:
        st.write('Interceptions/ball recoveries that lead to a chance (0,01 xg)')
        st.pyplot(fig)
    store_chancer_sequencer_spillere = store_chancer_sequencer.value_counts('playerName').reset_index()
    store_chancer_sequencer_spillere.columns = ['playerName', 'Number']

    with col2:
        st.write('Players with interceptions/ball recoveries')
        st.dataframe(store_chancer_sequencer_spillere,hide_index=True)
        

    st.title('Against ' + selected_opponent)
    #Modstanders modstandere
    df_possession_modstander = load_modstander_possession_data(selected_opponent)

    df_keypass = df_possession_modstander[df_possession_modstander['team_name'] != selected_opponent]
    df_keypass = df_keypass[df_keypass['label'].isin(Kampvalg)]
    df_keypass = df_keypass[df_keypass['q_qualifierId'] == 210.0]
    df_keypass = df_keypass.drop_duplicates('id')
    df_keypass_spiller = df_keypass['playerName'].value_counts()
    df_keypass_spiller = df_keypass_spiller.sort_values(ascending=False)

    df_xg_modstander = df_xg[df_xg['label'].str.contains(selected_opponent)]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['label'].isin(Kampvalg)]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['team_name'] != selected_opponent]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_modstander = df_xg_modstander[df_xg_modstander['q_value'].astype(float) > 0]
    df_xg_modstander['q_value'] = df_xg_modstander['q_value'].astype(float)
    df_xg_spiller = df_xg_modstander.groupby('playerName')['q_value'].sum()
    df_xg_spiller = df_xg_spiller.sort_values(ascending=False)


    df_assist = df_possession_modstander.copy()
    df_assist = df_assist[df_assist['label'].isin(Kampvalg)]
    df_assist = df_assist[df_assist['team_name'] != selected_opponent]
    df_assist = df_assist[df_assist['assist'] == 1]
    df_assist = df_assist.drop_duplicates('id')
    df_assist_spiller = df_assist.groupby('playerName')['assist'].sum()
    df_assist_spiller = df_assist_spiller.sort_values(ascending=False)


    df_possession_modstander = df_possession_modstander[df_possession_modstander['label'].isin(Kampvalg)]
    df_possession_modstander = df_possession_modstander[df_possession_modstander['team_name'] == selected_opponent]

    df_xg_plot = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot = df_xg_plot[df_xg_plot['q_value'].astype(float) > 0.0]

    col1,col2 = st.columns(2)
    x = df_xg_plot['x'].astype(float)
    y = df_xg_plot['y'].astype(float)
    shot_xg = df_xg_plot['q_value'].astype(float)
    player_names = df_xg_plot['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)

    with col1:
        st.write('Xg plot mod '+ selected_opponent)
        st.pyplot(fig)

    df_xg_plot_store_chancer = df_xg_modstander[df_xg_modstander['q_qualifierId'].astype(int) == 321]
    df_xg_plot_store_chancer = df_xg_plot_store_chancer[df_xg_plot_store_chancer['q_value'].astype(float) > 0.2]

    x = df_xg_plot_store_chancer['x'].astype(float)
    y = df_xg_plot_store_chancer['y'].astype(float)
    shot_xg = df_xg_plot_store_chancer['q_value'].astype(float)
    player_names = df_xg_plot_store_chancer['playerName'].astype(str)

    min_size = 1  # Minimum dot size
    max_size = 50  # Maximum dot size
    sizes = np.interp(shot_xg, (shot_xg.min(), shot_xg.max()), (min_size, max_size))

    pitch = Pitch(pitch_type='opta',half=True, pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax, s=sizes)
    with col2:
        st.write('Xg plot store chancer mod ' + selected_opponent)
        st.pyplot(fig)
    
    col1,col2 = st.columns(2)

    x = df_keypass['x'].astype(float)
    y = df_keypass['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    with col1:
        st.write('Shot assists mod ' + selected_opponent)
        st.pyplot(fig)

    x = df_assist['x'].astype(float)
    y = df_assist['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    with col2:
        st.write('Assists mod ' + selected_opponent)
        st.pyplot(fig)
        
    #sorterer for standardsituationer
    df_possession_modstander['q_qualifierId'] = df_possession_modstander['q_qualifierId'].astype(float)
    filtered_ids = df_possession_modstander[df_possession_modstander['q_qualifierId'].isin([22,23])]['id']
    # Filter out all rows with the filtered 'id' values
    filtered_data = df_possession_modstander[df_possession_modstander['id'].isin(filtered_ids)]
    #erobringer til store chancer
    df_store_chancer = filtered_data[(filtered_data['q_qualifierId'] == 321)]
    df_store_chancer = df_store_chancer[df_store_chancer['q_value'].astype(float) > 0.01]
    store_chancer_sequencer = df_store_chancer[['label','sequenceId']]
    store_chancer_sequencer = store_chancer_sequencer.merge(df_possession_modstander)
    store_chancer_sequencer = store_chancer_sequencer.drop_duplicates(subset='sequenceId', keep='first')
    x = store_chancer_sequencer['x'].astype(float)
    y = store_chancer_sequencer['y'].astype(float)

    pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw()
    sc = pitch.scatter(x, y, ax=ax)
    col1,col2 = st.columns(2)
    with col1:
        st.write('Erobringer der fører til chancer mod ' + selected_opponent + ' (0,01 xg)')
        st.pyplot(fig)
    store_chancer_sequencer_spillere = store_chancer_sequencer.value_counts('playerName')

st.cache_data(experimental_allow_widgets=True)
st.cache_resource(experimental_allow_widgets=True)
def League_stats():
    matchstats_columns = ['contestantId','label', 'date','attAssistSetplay','touches','totalLongBalls', 'duelLost', 'aerialLost', 'successfulOpenPlayPass', 'totalContest', 'duelWon', 'penAreaEntries', 'accurateBackZonePass', 'possWonDef3rd', 'wonContest', 'accurateFwdZonePass', 'openPlayPass', 'totalBackZonePass', 'minsPlayed', 'fwdPass', 'finalThirdEntries', 'ballRecovery', 'totalFwdZonePass', 'successfulFinalThirdPasses', 'totalFinalThirdPasses', 'attAssistOpenplay', 'aerialWon', 'totalAttAssist', 'possWonMid3rd', 'interception', 'totalCrossNocorner', 'interceptionWon', 'attOpenplay', 'touchesInOppBox', 'attemptsIbox', 'totalThroughBall', 'possWonAtt3rd', 'accurateCrossNocorner', 'bigChanceCreated', 'accurateThroughBall', 'totalLayoffs', 'accurateLayoffs', 'totalFastbreak', 'shotFastbreak']
    matchstats_df = pd.read_csv(r'1. Division/matchstats_all 1. Division.csv', usecols=matchstats_columns)
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
    
    xg_df = pd.read_csv(r'1. Division/xg_all 1. Division.csv')
    xg_df_openplay_id = xg_df[xg_df['q_qualifierId'].isin([6.0,9.0,26.0,25.0,24.0,107.0])]
    xg_df_openplay = xg_df[~xg_df['id'].isin(xg_df_openplay_id['id'])]
    xg_df_openplay = xg_df_openplay[xg_df_openplay['q_qualifierId'] == 321]
    xg_df_openplay['q_value'] = xg_df_openplay['q_value'].astype(float)

    xg_df_openplay = xg_df_openplay.groupby(['contestantId', 'team_name', 'date'])['q_value'].sum().reset_index()
    xg_df_openplay = xg_df_openplay.rename(columns={'q_value': 'open play xG'})
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

    cols_to_rank = matchstats_df.drop(columns=['contestantId', 'team_name','matches','minsPlayed']).columns
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
    'Match evaluation' : Match_evaluation,
    'Team development' : Team_development,
    'Opposition analysis' : Opposition_analysis,
    'League stats': League_stats
}

selected_data = st.sidebar.radio('Choose data type',list(Data_types.keys()))
Data_types[selected_data]()
