import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# prediction model
pipe = pickle.load(open('pipe (1).pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# Title and Design
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🏏 IPL Win Predictor</h1>
    <hr style='border: 2px solid #4CAF50;'>
    """,
    unsafe_allow_html=True
)

# Side Panel for Inputs
with st.sidebar:
    st.header("Match Details")
    batting_team = st.selectbox("Select Batting Team ", sorted(teams))
    bowling_team = st.selectbox("Select Bowling Team", sorted(teams))
    city = st.selectbox("Select the city where the match is being played", sorted(cities))
    target_score = st.number_input("Target Score")
    current_score = st.number_input("Current Score")
    
    overs = st.number_input("Overs Completed")
    wickets = st.number_input("Wickets Lost")
    runs_left = target_score-current_score
    balls_left = 120-(overs*6)
    wickets = 10-wickets
    if overs == 0:
        
        currentrunrate = 0
    else:
        currentrunrate = current_score / overs
    
    requiredrunrate = (runs_left*6)/balls_left



    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city], 'runs_left': [runs_left], 'balls_left': [
                            balls_left], 'wickets': [wickets], 'total_runs_x': [target_score], 'cur_run_rate': [currentrunrate], 'req_run_rate': [requiredrunrate]})


# Prediction
result = pipe.predict_proba(input_df)
win_prob = result[0][1]
remaining_prob = result[0][0]

# Display Win Probability with Plotly Gauge
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_prob*100,
        title={'text': f"Win Probability: {batting_team}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#4CAF50'},
            'steps': [
                {'range': [0, 40], 'color': '#F44336'},
                {'range': [40, 70], 'color': '#FF9800'},
                {'range': [70, 100], 'color': '#4CAF50'}
            ]
        }
    ))
    st.plotly_chart(fig)

with col2:
    fig_pie = go.Figure(data=[
        go.Pie(
            labels=[batting_team, bowling_team],
            values=[win_prob, remaining_prob],
            hole=0.4,
            marker=dict(colors=['#4CAF50', '#F44336']),
            textinfo='label+percent'
        )
    ])
    st.plotly_chart(fig_pie)

# Additional Match Insights
st.markdown("### 📊 Match Insights")
st.write(f"**Projected Run Rate:** {round(currentrunrate, 2)} runs/over")
st.write(f"**Required Run Rate:** {round(requiredrunrate, 2)} runs/over")

# Score Progression Chart
st.markdown("### 📈 Score Progression")
progress_df = pd.DataFrame({
    'Overs': np.arange(1, int(overs) + 1),
    'Runs': np.linspace(0, current_score, int(overs))
})

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=progress_df['Overs'],
    y=progress_df['Runs'],
    mode='lines+markers',
    line=dict(color='#4CAF50', width=3),
    marker=dict(size=8)
))

fig_line.update_layout(
    xaxis_title='Overs Completed',
    yaxis_title='Runs Scored',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=20, b=20)
)

st.plotly_chart(fig_line)

# Footer
st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)
st.markdown("✅ Developed with ❤️ using **Streamlit** and **Plotly**  By YUGANT")

