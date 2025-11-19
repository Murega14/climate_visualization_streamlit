import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import requests_cache
from retry_requests import retry
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Architectural Climate Analysis",
    page_icon="🏗️",
    layout="wide"
)

# Initialize cache session
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_interactive_wind_rose(wind_speeds, wind_directions, title="Wind Rose"):
    """
    Create an interactive wind rose using Plotly
    """
    # Define wind direction bins (16 directions)
    direction_bins = np.arange(0, 360, 22.5)
    direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                       'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Define wind speed bins (m/s)
    speed_bins = [0, 2, 5, 8, 11, 14, 17, 20]
    speed_labels = ['Calm (<2)', 'Light (2-5)', 'Moderate (5-8)', 'Fresh (8-11)',
                   'Strong (11-14)', 'Gale (14-17)', 'Storm (>17)']
    
    # Calculate frequencies
    direction_speed_data = []
    
    for i in range(len(direction_bins)):
        dir_start = direction_bins[i]
        dir_end = direction_bins[(i + 1) % len(direction_bins)]
        
        if i == len(direction_bins) - 1:
            mask = (wind_directions >= dir_start) | (wind_directions < dir_end)
        else:
            mask = (wind_directions >= dir_start) & (wind_directions < dir_end)
        
        dir_speeds = wind_speeds[mask]
        speed_counts = []
        
        for j in range(len(speed_bins) - 1):
            speed_mask = (dir_speeds >= speed_bins[j]) & (dir_speeds < speed_bins[j + 1])
            speed_counts.append(np.sum(speed_mask))
        
        direction_speed_data.append(speed_counts)
    
    total_count = len(wind_directions)
    direction_speed_percent = (np.array(direction_speed_data) / total_count) * 100
    
    # Create plotly figure
    fig = go.Figure()
    
    colors = px.colors.sequential.Plasma
    
    for speed_idx in range(len(speed_labels)):
        radii = direction_speed_percent[:, speed_idx]
        theta = direction_bins
        
        fig.add_trace(go.Barpolar(
            r=radii,
            theta=theta,
            width=22.5,
            name=speed_labels[speed_idx],
            marker_color=colors[speed_idx],
            marker_line_color='white',
            marker_line_width=1,
            opacity=0.8
        ))
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, np.max(direction_speed_percent.sum(axis=1)) * 1.1]
            ),
            angularaxis=dict(
                direction='clockwise',
                period=360
            )
        ),
        showlegend=True,
        legend=dict(
            title="Wind Speed (m/s)",
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        ),
        height=600
    )
    
    return fig

def create_interactive_climate_chart(monthly_profile):
    """
    Create interactive temperature and rainfall chart using Plotly
    """
    months = list(monthly_profile.keys())
    temp_data = [monthly_profile[month]['temperature_c'] for month in months]
    rain_data = [monthly_profile[month]['rainfall_mm'] for month in months]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add temperature line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=temp_data,
            name="Temperature",
            mode='lines+markers',
            line=dict(color='rgb(255, 127, 14)', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ),
        secondary_y=False
    )
    
    # Add rainfall bars
    fig.add_trace(
        go.Bar(
            x=months,
            y=rain_data,
            name="Rainfall",
            marker_color='rgb(31, 119, 180)',
            opacity=0.7
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, color='rgb(255, 127, 14)')
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, color='rgb(31, 119, 180)')
    
    fig.update_layout(
        title="Monthly Temperature & Rainfall Profile",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_interactive_solar_chart(solar_df):
    """
    Create interactive solar radiation chart using Plotly
    """
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=solar_df['date'],
        y=solar_df['radiation_mj_m2_day'],
        mode='lines+markers',
        name='Solar Radiation',
        line=dict(color='rgb(255, 165, 0)', width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.3)'
    ))
    
    # Add average line
    avg_solar = solar_df['radiation_mj_m2_day'].mean()
    fig.add_hline(
        y=avg_solar,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {avg_solar:.1f} MJ/m²/day",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Monthly Solar Radiation Analysis",
        xaxis_title="Date",
        yaxis_title="Radiation (MJ/m²/day)",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_climate_heatmap(monthly_profile):
    """
    Create enhanced heatmap visualization for climate data
    """
    months = list(monthly_profile.keys())
    month_abbr = [m[:3] for m in months]
    temp_data = [monthly_profile[month]['temperature_c'] for month in months]
    rain_data = [monthly_profile[month]['rainfall_mm'] for month in months]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Temperature heatmap
    temp_matrix = np.array([temp_data])
    im1 = ax1.imshow(temp_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    ax1.set_xticks(np.arange(len(months)))
    ax1.set_xticklabels(month_abbr, fontsize=11, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_title('Monthly Temperature Profile (°C)', fontsize=14, fontweight='bold', pad=15)
    
    # Add temperature values
    for i, temp in enumerate(temp_data):
        color = 'white' if temp > (max(temp_data) + min(temp_data))/2 else 'black'
        ax1.text(i, 0, f'{temp:.1f}°C', ha='center', va='center', 
                fontweight='bold', fontsize=10, color=color)
    
    plt.colorbar(im1, ax=ax1, orientation='vertical', pad=0.02, label='Temperature (°C)')
    
    # Rainfall heatmap
    rain_matrix = np.array([rain_data])
    im2 = ax2.imshow(rain_matrix, cmap='Blues', aspect='auto', interpolation='bilinear')
    ax2.set_xticks(np.arange(len(months)))
    ax2.set_xticklabels(month_abbr, fontsize=11, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_title('Monthly Rainfall Profile (mm)', fontsize=14, fontweight='bold', pad=15)
    
    # Add rainfall values
    for i, rain in enumerate(rain_data):
        color = 'white' if rain > max(rain_data)*0.6 else 'black'
        ax2.text(i, 0, f'{rain:.0f}', ha='center', va='center', 
                fontweight='bold', fontsize=10, color=color)
    
    plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02, label='Rainfall (mm)')
    
    plt.tight_layout()
    return fig

def get_solar_from_openmeteo(lat: float, lon: float) -> pd.DataFrame:
    """Fetch monthly solar radiation data"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "daily": "shortwave_radiation_sum",
            "timezone": "auto"
        }

        response = retry_session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        daily_data = data.get('daily', {})
        dates = pd.to_datetime(daily_data.get('time', []))
        radiation_values = daily_data.get('shortwave_radiation_sum', [])
        
        if len(dates) == 0 or len(radiation_values) == 0:
            return None
        
        # Convert to MJ/m²
        if np.mean(radiation_values) < 1000:
            daily_df = pd.DataFrame({
                "date": dates,
                "radiation_mj_m2": radiation_values
            })
        else:
            daily_df = pd.DataFrame({
                "date": dates,
                "radiation_mj_m2": [x / 1_000_000 for x in radiation_values]
            })
        
        # Resample to monthly
        monthly_df = daily_df.resample('M', on='date').agg({
            'radiation_mj_m2': 'sum'
        }).reset_index()
        
        monthly_df['days_in_month'] = monthly_df['date'].dt.days_in_month
        monthly_df['radiation_mj_m2_day'] = monthly_df['radiation_mj_m2'] / monthly_df['days_in_month']
        
        return monthly_df

    except Exception as e:
        st.error(f"Open-Meteo solar data error: {e}")
        return None

def get_monthly_climate_profile(lat: float, lon: float, start_year: int = 2014, end_year: int = 2024):
    """
    Fetch monthly average climate profile from NASA POWER
    """
    NASA_BASE = "https://power.larc.nasa.gov/api"
    
    url = f"{NASA_BASE}/temporal/climatology/point"
    params = {
        "start": start_year,
        "end": end_year,
        "latitude": lat,
        "longitude": lon,
        "community": "ag",
        "parameters": "PRECTOTCORR,T2M,ALLSKY_SFC_SW_DWN",
        "format": "json",
        "units": "metric",
        "header": "true"
    }

    try:
        r = retry_session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        parameters = data.get("properties", {}).get("parameter", {})
        rainfall = parameters.get("PRECTOTCORR", {})
        temperature = parameters.get("T2M", {})
        solar = parameters.get("ALLSKY_SFC_SW_DWN", {})

        if not rainfall or not temperature:
            return None

        month_names = {
            "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
            "MAY": "May", "JUN": "June", "JUL": "July", "AUG": "August",
            "SEP": "September", "OCT": "October", "NOV": "November", "DEC": "December"
        }

        monthly_profile = {}
        for m_code, m_name in month_names.items():
            monthly_profile[m_name] = {
                "rainfall_mm": round(rainfall.get(m_code, 0), 2),
                "temperature_c": round(temperature.get(m_code, 0), 2),
            }

        return monthly_profile

    except Exception as e:
        st.error(f"NASA POWER API error: {e}")
        return None

def get_wind_data_from_api(lat, lon):
    """
    Get wind data from Open-Meteo API
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }
        
        response = retry_session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = data.get('hourly', {})
        dates = pd.to_datetime(hourly_data.get('time', []))
        wind_speeds = hourly_data.get('wind_speed_10m', [])
        wind_directions = hourly_data.get('wind_direction_10m', [])
        
        if len(dates) == 0:
            return None
            
        wind_df = pd.DataFrame({
            "date": dates,
            "wind_speed": wind_speeds,
            "wind_direction": wind_directions
        })
        
        # Remove null values
        wind_df = wind_df.dropna()
        
        return wind_df
        
    except Exception as e:
        st.error(f"Error fetching wind data: {e}")
        return None

def analyze_rainfall_patterns(monthly_profile):
    """
    Analyze rainfall patterns and provide architectural insights
    """
    months = list(monthly_profile.keys())
    rainfall_data = [monthly_profile[month]['rainfall_mm'] for month in months]
    
    total_annual_rainfall = sum(rainfall_data)
    max_rainfall = max(rainfall_data)
    min_rainfall = min(rainfall_data)
    wettest_month = months[rainfall_data.index(max_rainfall)]
    driest_month = months[rainfall_data.index(min_rainfall)]
    
    # Determine rainfall regime
    if total_annual_rainfall > 1500:
        rainfall_regime = "High Rainfall"
        roof_slope = "Steep (>30°)"
        drainage = "High capacity required"
    elif total_annual_rainfall > 800:
        rainfall_regime = "Moderate Rainfall"
        roof_slope = "Moderate (15-30°)"
        drainage = "Standard capacity"
    else:
        rainfall_regime = "Low Rainfall"
        roof_slope = "Gentle (<15°)"
        drainage = "Basic capacity"
    
    rainy_months = [month for month, rain in zip(months, rainfall_data) if rain > 100]
    dry_months = [month for month, rain in zip(months, rainfall_data) if rain < 50]
    
    if len(rainy_months) >= 6:
        seasonality = "Year-round rainfall"
    elif len(rainy_months) >= 3:
        seasonality = "Seasonal rainfall"
    else:
        seasonality = "Distinct dry season"
    
    return {
        'total_annual_rainfall': total_annual_rainfall,
        'max_rainfall': max_rainfall,
        'min_rainfall': min_rainfall,
        'wettest_month': wettest_month,
        'driest_month': driest_month,
        'rainfall_regime': rainfall_regime,
        'roof_slope': roof_slope,
        'drainage': drainage,
        'seasonality': seasonality,
        'rainy_months': rainy_months,
        'dry_months': dry_months,
        'monthly_rainfall': dict(zip(months, rainfall_data))
    }

def create_architectural_climate_dashboard():
    st.title("Architectural Climate Analysis Dashboard")
    st.markdown("*Professional climate analysis for sustainable architectural design*")
    
    query_params = st.query_params
    
    url_lat = query_params.get('lat')
    url_lon = query_params.get('lon')
    
    default_lat = float(url_lat if url_lat else -1.0388)
    default_lon = float(url_lon if url_lon else 37.0834)
    
    auto_analyze = url_lat is not None and url_lon is not None
    
    # Sidebar
    st.sidebar.header("Location Settings")
    lat = st.sidebar.number_input("Latitude", value=default_lat, format="%.4f", 
                                   help="Enter latitude in decimal degrees")
    lon = st.sidebar.number_input("Longitude", value=default_lon, format="%.4f", 
                                   help="Enter longitude in decimal degrees")
    
    st.sidebar.header("Analysis Period")
    start_year = st.sidebar.number_input("Start Year", value=2014, min_value=1980, max_value=2023)
    end_year = st.sidebar.number_input("End Year", value=2024, min_value=1981, max_value=2024)
    
    st.sidebar.header("Display Settings")
    show_insights = st.sidebar.checkbox("Show Detailed Insights", value=True)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    # Auto-trigger analysis if URL params exist and data not loaded
    if auto_analyze and not st.session_state.get('data_loaded', False):
        with st.spinner(f"Fetching climate data for {location_name}..."):
            monthly_profile = get_monthly_climate_profile(lat, lon, start_year, end_year)
            solar_df = get_solar_from_openmeteo(lat, lon)
            wind_df = get_wind_data_from_api(lat, lon)
        
        if monthly_profile is not None:
            st.session_state['monthly_profile'] = monthly_profile
            st.session_state['solar_df'] = solar_df
            st.session_state['wind_df'] = wind_df
            st.session_state['location_name'] = location_name
            st.session_state['data_loaded'] = True
        else:
            st.error("Unable to fetch climate data from NASA POWER. Please check your internet connection and coordinates.")
            return
    
    # Manual analyze button
    if st.sidebar.button("Analyze Climate Data", type="primary"):
        with st.spinner(f"Fetching climate data for {location_name}..."):
            monthly_profile = get_monthly_climate_profile(lat, lon, start_year, end_year)
            solar_df = get_solar_from_openmeteo(lat, lon)
            wind_df = get_wind_data_from_api(lat, lon)
        
        if monthly_profile is None:
            st.error("Unable to fetch climate data from NASA POWER. Please check your internet connection and coordinates.")
            return
        
        if solar_df is None or solar_df.empty:
            st.warning("Solar radiation data unavailable. Some visualizations will be limited.")
        
        if wind_df is None or wind_df.empty:
            st.warning("Wind data unavailable. Wind analysis will be limited.")
        
        st.session_state['monthly_profile'] = monthly_profile
        st.session_state['solar_df'] = solar_df
        st.session_state['wind_df'] = wind_df
        st.session_state['location_name'] = location_name
        st.session_state['data_loaded'] = True
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("Click 'Analyze Climate Data' to fetch and display climate information for your location.")
        return
    
    # Retrieve from session state
    monthly_profile = st.session_state['monthly_profile']
    solar_df = st.session_state['solar_df']
    wind_df = st.session_state['wind_df']
    location_name = st.session_state['location_name']
    
    # Process temperature data
    months = list(monthly_profile.keys())
    avg_temps = [monthly_profile[month]['temperature_c'] for month in months]
    
    max_temp = max(avg_temps)
    hottest_month = months[avg_temps.index(max_temp)]
    min_temp = min(avg_temps)
    coldest_month = months[avg_temps.index(min_temp)]
    avg_annual_temp = sum(avg_temps) / len(avg_temps)
    
    # Analyze rainfall patterns
    rainfall_analysis = analyze_rainfall_patterns(monthly_profile)
    
    # Process wind data if available
    if wind_df is not None and not wind_df.empty:
        avg_wind_speed = wind_df['wind_speed'].mean()
        prevailing_direction = wind_df['wind_direction'].mode().iloc[0] if not wind_df['wind_direction'].mode().empty else 0
        wind_speeds = wind_df['wind_speed'].values
        wind_directions = wind_df['wind_direction'].values
    else:
        avg_wind_speed = None
        prevailing_direction = None
        wind_speeds = None
        wind_directions = None
    
    # Main dashboard layout
    st.subheader(f"Climate Summary for {location_name}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hottest Month", f"{hottest_month}", f"{max_temp:.1f}°C")
        st.metric("Coldest Month", f"{coldest_month}", f"{min_temp:.1f}°C")
        
    with col2:
        st.metric("Annual Avg Temperature", f"{avg_annual_temp:.1f}°C")
        st.metric("Temperature Range", f"{max_temp - min_temp:.1f}°C")
        
    with col3:
        if avg_wind_speed is not None:
            st.metric("Average Wind Speed", f"{avg_wind_speed:.1f} m/s")
            st.metric("Prevailing Wind", f"{prevailing_direction:.0f}°")
        else:
            st.metric("Average Wind Speed", "N/A")
            st.metric("Prevailing Wind", "N/A")
        
    with col4:
        st.metric("Annual Rainfall", f"{rainfall_analysis['total_annual_rainfall']:.0f} mm")
        st.metric("Rainfall Regime", rainfall_analysis['rainfall_regime'])
    
    # Interactive climate chart
    st.subheader("Interactive Climate Overview")
    fig_climate = create_interactive_climate_chart(monthly_profile)
    st.plotly_chart(fig_climate, use_container_width=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Rainfall Analysis", "Climate Patterns", "Solar Analysis", "Wind Analysis"])
    
    with tab1:
        st.write("**Monthly Rainfall Distribution**")
        
        # Create rainfall bar chart
        months_list = list(rainfall_analysis['monthly_rainfall'].keys())
        rainfall_values = list(rainfall_analysis['monthly_rainfall'].values())
        
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=[m[:3] for m in months_list],
            y=rainfall_values,
            marker_color=rainfall_values,
            marker_colorscale='Blues',
            text=[f'{v:.0f}mm' for v in rainfall_values],
            textposition='outside'
        ))
        
        avg_rainfall = np.mean(rainfall_values)
        fig_rain.add_hline(
            y=avg_rainfall,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_rainfall:.1f}mm"
        )
        
        fig_rain.update_layout(
            title="Monthly Rainfall Distribution",
            xaxis_title="Month",
            yaxis_title="Rainfall (mm)",
            height=500
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        
        # Rainfall insights
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""**Rainfall Patterns:**
- Wettest: {rainfall_analysis['wettest_month']} ({rainfall_analysis['max_rainfall']:.0f}mm)
- Driest: {rainfall_analysis['driest_month']} ({rainfall_analysis['min_rainfall']:.0f}mm)
- Seasonality: {rainfall_analysis['seasonality']}
- Total Annual: {rainfall_analysis['total_annual_rainfall']:.0f}mm""")
        
        with col2:
            roof_area = st.slider("Roof Area (m²) for Harvesting", 50, 500, 100)
            potential_harvest = (rainfall_analysis['total_annual_rainfall'] / 1000) * roof_area * 0.8
            st.success(f"""**Rainwater Harvesting:**
- Potential: {potential_harvest:.0f} m³/year
- For {roof_area}m² roof (80% efficiency)
- Roof slope: {rainfall_analysis['roof_slope']}
- Drainage: {rainfall_analysis['drainage']}""")
    
    with tab2:
        st.write("**Climate Heatmap Analysis**")
        fig_heat = create_climate_heatmap(monthly_profile)
        st.pyplot(fig_heat)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""**Temperature Insights:**
- Annual average: {avg_annual_temp:.1f}°C
- Seasonal variation: {max_temp - min_temp:.1f}°C
- Hottest: {hottest_month} ({max_temp:.1f}°C)
- Coldest: {coldest_month} ({min_temp:.1f}°C)""")
        
        with col2:
            if max_temp > 30:
                comfort_level = "Hot climate"
                strategies = "Shading, ventilation, thermal mass, light colors"
            elif min_temp < 10:
                comfort_level = "Cool climate"
                strategies = "Insulation, solar gain, thermal mass"
            else:
                comfort_level = "Temperate climate"
                strategies = "Mixed-mode ventilation, moderate insulation"
            
            st.success(f"""**Design Recommendations:**
- Climate: {comfort_level}
- Key strategies: {strategies}
- Thermal comfort: Priority consideration""")
    
    with tab3:
        if solar_df is not None and not solar_df.empty:
            st.write("**Solar Radiation Analysis**")
            
            fig_solar = create_interactive_solar_chart(solar_df)
            st.plotly_chart(fig_solar, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                best_month = solar_df.loc[solar_df['radiation_mj_m2_day'].idxmax()]
                worst_month = solar_df.loc[solar_df['radiation_mj_m2_day'].idxmin()]
                avg_solar = solar_df['radiation_mj_m2_day'].mean()
                
                st.metric("Peak Solar", f"{best_month['radiation_mj_m2_day']:.1f} MJ/m²/day",
                         best_month['date'].strftime('%B %Y'))
                st.metric("Lowest Solar", f"{worst_month['radiation_mj_m2_day']:.1f} MJ/m²/day",
                         worst_month['date'].strftime('%B %Y'))
                st.metric("Annual Average", f"{avg_solar:.1f} MJ/m²/day")
            
            with col2:
                if avg_solar > 20:
                    solar_potential = "Excellent"
                    pv_efficiency = "High efficiency expected"
                elif avg_solar > 15:
                    solar_potential = "Good"
                    pv_efficiency = "Good efficiency expected"
                else:
                    solar_potential = "Moderate"
                    pv_efficiency = "Moderate efficiency expected"
                
                st.info(f"""**Solar Design:**
- Solar potential: {solar_potential}
- PV performance: {pv_efficiency}
- Peak month: {best_month['date'].strftime('%B')}
- Orientation: Optimize for equatorial position""")
        else:
            st.warning("Solar radiation data unavailable")
    
    with tab4:
        if wind_speeds is not None and wind_directions is not None:
            st.write("**Wind Analysis & Building Orientation**")
            
            fig_wind = create_interactive_wind_rose(wind_speeds, wind_directions, 
                                                    f"Wind Distribution - {location_name}")
            st.plotly_chart(fig_wind, use_container_width=True)
            
            # Determine cardinal direction
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                         'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            idx = int((prevailing_direction + 11.25) / 22.5) % 16
            cardinal = directions[idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prevailing Direction", f"{cardinal} ({prevailing_direction:.0f}°)")
                st.metric("Average Speed", f"{avg_wind_speed:.1f} m/s")
                
                max_wind = wind_df['wind_speed'].max()
                st.metric("Maximum Speed", f"{max_wind:.1f} m/s")
            
            with col2:
                if avg_wind_speed < 3:
                    wind_impact = "Minimal wind impact"
                    ventilation = "May need mechanical ventilation"
                    structure = "Standard construction"
                elif avg_wind_speed < 6:
                    wind_impact = "Good natural ventilation"
                    ventilation = "Optimize window placement for cross-ventilation"
                    structure = "Standard construction adequate"
                else:
                    wind_impact = "Significant wind forces"
                    ventilation = "Excellent natural ventilation, consider wind breaks"
                    structure = "Reinforced structure recommended"
                
                st.success(f"""**Wind Design:**
- Impact: {wind_impact}
- Ventilation: {ventilation}
- Structure: {structure}
- Orient main openings: {cardinal} facing""")
        else:
            st.warning("Wind data unavailable")
    
    # Detailed Insights Section
    if show_insights:
        st.subheader("Detailed Architectural Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.write("**limate Classification**")
            
            # Climate classification
            if avg_annual_temp > 18 and rainfall_analysis['total_annual_rainfall'] > 1000:
                climate_type = "Tropical"
                building_style = "Open, ventilated design"
                energy_strategy = "Cooling priority, natural ventilation"
            elif avg_annual_temp > 10 and rainfall_analysis['total_annual_rainfall'] > 500:
                climate_type = "Temperate"
                building_style = "Balanced design"
                energy_strategy = "Mixed-mode, seasonal adaptation"
            elif rainfall_analysis['total_annual_rainfall'] < 250:
                climate_type = "Arid"
                building_style = "Thermal mass, small openings"
                energy_strategy = "Passive cooling, shade"
            else:
                climate_type = "Cool Temperate"
                building_style = "Insulated, compact"
                energy_strategy = "Heating priority, solar gain"
            
            st.info(f"""**Classification:**
- Type: {climate_type}
- Building style: {building_style}
- Energy strategy: {energy_strategy}""")
        
        with insight_col2:
            st.write("**Construction Guidance**")
            
            # Wall construction
            if max_temp > 30 or min_temp < 5:
                wall_type = "Insulated walls required"
                wall_material = "Thermal mass materials (concrete, brick)"
            else:
                wall_type = "Standard construction"
                wall_material = "Various materials suitable"
            
            # Roofing
            if rainfall_analysis['total_annual_rainfall'] > 1000:
                roofing = "Durable waterproofing essential"
                roof_material = "Metal, tiles with underlayment"
            else:
                roofing = "Standard roofing adequate"
                roof_material = "Various materials suitable"
            
            st.info(f"""**Construction:**
- Walls: {wall_type}
- Material: {wall_material}
- Roofing: {roofing}
- Roof type: {roof_material}
- Slope: {rainfall_analysis['roof_slope']}""")
        
        with insight_col3:
            st.write("***Sustainability Features***")
            
            # Solar feasibility
            if solar_df is not None and not solar_df.empty:
                avg_solar = solar_df['radiation_mj_m2_day'].mean()
                if avg_solar > 20:
                    solar_feasibility = "Excellent - PV highly recommended"
                    solar_percentage = "Can meet 80-100% needs"
                elif avg_solar > 15:
                    solar_feasibility = "Good - PV recommended"
                    solar_percentage = "Can meet 50-80% needs"
                else:
                    solar_feasibility = "Moderate - PV viable"
                    solar_percentage = "Can meet 30-50% needs"
            else:
                solar_feasibility = "Data unavailable"
                solar_percentage = "Assessment pending"
            
            # Water management
            if rainfall_analysis['total_annual_rainfall'] > 800:
                water_strategy = "Rainwater harvesting essential"
            else:
                water_strategy = "Supplementary water needed"
            
            st.success(f"""**Sustainability:**
- Solar: {solar_feasibility}
- Potential: {solar_percentage}
- Water: {water_strategy}
- Ventilation: Natural preferred""")
    
    # Additional design recommendations
    st.subheader("Design Recommendations Summary")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.write("**Fenestration**")
        if max_temp > 28:
            window_rec = "Small windows on hot facades, large on cool sides"
            shading_rec = "External shading essential (overhangs, louvers)"
            glazing_rec = "Low-E glazing, tinted or reflective"
        elif min_temp < 12:
            window_rec = "Moderate windows, maximize solar gain"
            shading_rec = "Minimal shading, allow winter sun"
            glazing_rec = "Double glazing for insulation"
        else:
            window_rec = "Flexible window sizing, optimize for views"
            shading_rec = "Adjustable shading recommended"
            glazing_rec = "Standard or low-E glazing"
        
        st.markdown(f"""
- **Windows:** {window_rec}
- **Shading:** {shading_rec}
- **Glazing:** {glazing_rec}
""")
    
    with rec_col2:
        st.write("**Ventilation Strategy**")
        if avg_wind_speed and avg_wind_speed > 4:
            vent_strategy = "Cross-ventilation highly effective"
            vent_design = "Align openings with prevailing winds"
            mech_vent = "Minimal mechanical ventilation needed"
        elif avg_wind_speed and avg_wind_speed > 2:
            vent_strategy = "Natural ventilation viable"
            vent_design = "Stack effect and cross-ventilation"
            mech_vent = "Hybrid system recommended"
        else:
            vent_strategy = "Limited natural ventilation"
            vent_design = "Rely on stack effect"
            mech_vent = "Mechanical ventilation primary"
        
        st.markdown(f"""
- **Strategy:** {vent_strategy}
- **Design:** {vent_design}
- **Mechanical:** {mech_vent}
""")
    
    with rec_col3:
        st.write("**Envelope Design**")
        if max_temp - min_temp > 15:
            envelope = "High thermal mass for temperature swing"
            insulation = "Moderate to high insulation"
            color = "Light colors in hot season areas"
        elif max_temp > 30:
            envelope = "Minimize heat gain"
            insulation = "Insulation with reflective barriers"
            color = "Light, reflective colors"
        else:
            envelope = "Balanced thermal performance"
            insulation = "Standard insulation levels"
            color = "Color flexibility"
        
        st.markdown(f"""
- **Envelope:** {envelope}
- **Insulation:** {insulation}
- **Colors:** {color}
""")
    
    # Raw Data Section
    if show_raw_data:
        st.subheader(" Raw Climate Data")
        
        monthly_data = []
        for month in monthly_profile:
            monthly_data.append({
                'Month': month,
                'Temperature (°C)': monthly_profile[month]['temperature_c'],
                'Rainfall (mm)': monthly_profile[month]['rainfall_mm']
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Monthly Climate Data**")
            st.dataframe(monthly_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Climate Statistics**")
            stats_data = {
                'Metric': [
                    'Annual Average Temperature',
                    'Temperature Range',
                    'Total Annual Rainfall',
                    'Average Wind Speed',
                    'Solar Radiation Average'
                ],
                'Value': [
                    f"{avg_annual_temp:.1f}°C",
                    f"{max_temp - min_temp:.1f}°C",
                    f"{rainfall_analysis['total_annual_rainfall']:.0f} mm",
                    f"{avg_wind_speed:.1f} m/s" if avg_wind_speed else 'N/A',
                    f"{solar_df['radiation_mj_m2_day'].mean():.1f} MJ/m²/day" if solar_df is not None and not solar_df.empty else 'N/A'
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Export data option
        st.download_button(
            label=" Download Climate Data (CSV)",
            data=monthly_df.to_csv(index=False),
            file_name=f"{location_name}_climate_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong> Architectural Climate Analysis Dashboard</strong></p>
        <p>Data Sources: NASA POWER & Open-Meteo APIs | Built with Streamlit & Plotly</p>
        <p><em>For professional architectural and engineering applications</em></p>
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    create_architectural_climate_dashboard()