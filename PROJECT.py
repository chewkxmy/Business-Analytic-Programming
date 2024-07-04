import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import certifi
import folium
import pandas as pd
import plost
import pytz
import requests
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from streamlit_folium import st_folium

st.session_state.validity = False

if 'page' not in st.session_state:
    st.session_state.page = 0

pages = ['Home', 'Forecast', 'Flight']


def next_page():
    st.session_state.page += 1


def prev_page():
    st.session_state.page -= 1


def fetch_historical_weather(lat, long, api_key, date):
    url = f'https://api.weatherapi.com/v1/history.json?key={api_key}&q={lat},{long}&dt={date}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for date {date}: {e}')
        return None


def get_lat_long(location, api_key):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for geocoding {location}: {e}')
        return None, None


def get_api_key(service_name):
    api_keys = {
        'GEOCODE API': '3f34588e34254d84b706dae3e247814a',
        'WEATHER API': '65ade77659214036bec132718240107',
        'FLIGHT API': 'aWrrXWTDigGPGzMMqVIh3VVZnJVVHBEh',
        'SECRET API': 'fWn1JsgRWHvGjBsq',
    }
    return api_keys.get(service_name)


def connect_to_cassandra():
    try:
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect('sqit3073')
        return session
    except Exception as e:
        st.error(f'Failed to connect to Cassandra: {e}')
        return None


def check_existing_data(session, place, date):
    query = 'SELECT COUNT(*) FROM WEATHER WHERE place = %s AND date = %s'
    result = session.execute(query, (place, date)).one()
    return result.count > 0


def fetch_and_store_weather_data(session, lat, long, api_key, date, place):
    if check_existing_data(session, place, date):
        return None
    weather_data = fetch_historical_weather(lat, long, api_key, date)
    if weather_data and 'forecast' in weather_data and 'forecastday' in weather_data['forecast']:
        forecast_day = weather_data['forecast']['forecastday'][0]
        date = forecast_day['date']
        temperature = forecast_day['day']['avgtemp_c']
        weather_condition = forecast_day['day']['condition']['text']
        humidity = forecast_day['day']['avghumidity']
        windspeed = forecast_day['day']['maxwind_kph']
        place = weather_data['location']['name']
        result = (place, date, humidity, temperature, weather_condition, windspeed)

        if len(result) == 6:
            return result
    return None


def fetch_weather_data(session, place, start_date, end_date):
    query = '''
    SELECT * FROM weather
    WHERE place = %s AND date >= %s AND date <= %s
    '''
    try:
        result = session.execute(query, (place, str(start_date), str(end_date)))
        return pd.DataFrame(list(result))
    except Exception as e:
        print(f'Error executing query: {e}')
        return pd.DataFrame()


def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['weather_condition'] = label_encoder.fit_transform(df['weather_condition'])
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(f'Class mapping: {class_mapping}')
    X = df[['temperature', 'humidity', 'windspeed']]
    y = df['weather_condition']

    data = pd.concat([X, y], axis=1)

    max_size = data['weather_condition'].value_counts().max()
    lst = [data]
    for class_index, group in data.groupby('weather_condition'):
        lst.append(group.sample(max_size - len(group), replace=True))
    data_resampled = pd.concat(lst)

    X_resampled = data_resampled.drop('weather_condition', axis=1)
    y_resampled = data_resampled['weather_condition']

    return X_resampled, y_resampled, label_encoder


def train_model(X, y):
    print(len(X))
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(len(X_train))  # Number of training samples
    print(len(X_test))

    model = DecisionTreeClassifier(criterion='gini', random_state=42, splitter='best')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation(y_test, y_pred)

    return model, scaler, accuracy, precision


def print_evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))


def fetch_forecast_data(place, api_key):
    url = f'https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={place}&days=7'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for forecast data: {e}')
        return None


def make_forecasts(place, api_key):
    forecast_data = fetch_forecast_data(place, api_key)
    if not forecast_data:
        return [], [], [], [], []
    forecast_days = forecast_data['forecast']['forecastday']
    forecast_dates = []
    forecast_conditions = []
    temperature_data = []
    humidity_data = []
    windspeed_data = []

    for day in forecast_days:
        date = day['date']
        temperature = day['day']['avgtemp_c']
        humidity = day['day']['avghumidity']
        windspeed = day['day']['maxwind_kph']
        weather_condition = day['day']['condition']['text']

        forecast_dates.append(date)
        forecast_conditions.append(weather_condition)
        temperature_data.append(temperature)
        humidity_data.append(humidity)
        windspeed_data.append(windspeed)

    return forecast_dates, forecast_conditions, temperature_data, humidity_data, windspeed_data


def should_visit(forecast_conditions):
    bad_weather_keywords = ['cloudy', 'rain', 'rainy', 'thunderstorm', 'snow', 'sleet']
    bad_weather_count = sum(
        1 for weather in forecast_conditions if any(keyword in weather.lower() for keyword in bad_weather_keywords))

    if bad_weather_count >= 4:
        return 'It is not recommended to visit this place due to bad weather conditions on multiple days.'
    else:
        return 'The weather looks good! It is recommended to visit this place.'


# Mock function to fetch flight details
def get_flight_offers(origin, destination, departure_date, access_token):
    url = 'https://test.api.amadeus.com/v2/shopping/flight-offers'
    params = {
        'originLocationCode': origin,
        'destinationLocationCode': destination,
        'departureDate': departure_date,
        'adults': 1,  # Number of adult passengers
        'currencyCode': 'MYR'  # Currency for prices
    }
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    return data


image_mapping = {
    'sunny': '/Users/mchalxz/Downloads/sun.png',
    'rainy': '/Users/mchalxz/Downloads/rainy-day.png',
    'partly cloudy': '/Users/mchalxz/Downloads/cloud.png',
    'cloudy': '/Users/mchalxz/Downloads/cloudy.png',
    'heavy rain': '/Users/mchalxz/Downloads/heavy-rain-2.png',
    'patchy rain': '/Users/mchalxz/Downloads/nature.png',
    'rain': '/Users/mchalxz/Downloads/heavy-rain.png',
    'wind': '/Users/mchalxz/Downloads/wind.png'
}


def get_image_path(condition):
    condition_lower = condition.lower()
    for keyword, image_path in image_mapping.items():
        if keyword in condition_lower:
            return image_path
    return '/Users/mchalxz/PycharmProjects/SQIT3073/test/default.jpg'


def get_access_token(api_key, secret_key):
    url = 'https://test.api.amadeus.com/v1/security/oauth2/token'
    data = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': secret_key
    }

    try:
        response = requests.post(url, data=data, verify=False)
        response.raise_for_status()
        access_token = response.json().get('access_token')
        print(access_token)
        return access_token
    except requests.exceptions.SSLError as e:
        print(f"SSL error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


st.markdown(
    '''
    <div style='display: flex; align-items: center; justify-content: center;'>
        <span style='font-size: 2em; color: green; margin-right: 10px;'>🌤️</span>
        <h1 style='color: green; margin: 0 10px; text-align: center;'>
            Vacation Decision Maker with <br> Weather Forecasting
        </h1>
        <span style='font-size: 2em; color: green; margin-left: 10px;'>🌧️</span>
    </div>
    ''',
    unsafe_allow_html=True
)

if st.session_state.page == 0:

    rows = connect_to_cassandra().execute('SELECT location FROM iata')
    destinations = [(row.location) for row in rows]

    # Creating a list of destinations for the combobox
    destination_list = [f'{city}' for city in destinations]
    destination_list.insert(0, 'None')
    selected_destination = st.selectbox('Select a destination', destination_list)
    if selected_destination == 'None':
        st.error('Please select a destination.')
        st.stop()
    if selected_destination != 'None':
        lat, lon = get_lat_long(selected_destination, get_api_key('GEOCODE API'))

        my_map = folium.Map(location=[lat, lon], zoom_start=12)

        # Add a marker at the location
        folium.Marker([lat, lon], popup='Selected Location').add_to(my_map)

        # Display the map in Streamlit
        st_folium(my_map, width=700, height=500)

    st.session_state.place = selected_destination

    process = st.button('Continue')

    if process:
        with st.spinner('Processing...'):
            lat, long = get_lat_long(selected_destination, get_api_key('GEOCODE API'))
            if lat is None or long is None:
                st.error(f'Failed to get latitude and longitude for {selected_destination}.')
            else:
                end_date_obj = datetime.date.today()
                start_date_obj = end_date_obj - datetime.timedelta(days=365)
                session = connect_to_cassandra()
                if session is None:
                    st.error('Failed to connect to Cassandra. Exiting.')
                else:
                    dates = [(start_date_obj + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(365)]
                    results = []

                    with ThreadPoolExecutor(max_workers=10) as executor:
                        future_to_date = {executor.submit(fetch_and_store_weather_data, session, lat, long,
                                                          get_api_key('WEATHER API'), date, selected_destination): date
                                          for date in
                                          dates}
                        for future in as_completed(future_to_date):
                            data = future.result()
                            if data:
                                results.append(data)

                    # Prepare statement and batch insert into Cassandra
                    insert_query = 'INSERT INTO WEATHER (place, date, humidity, temperature, weather_condition, windspeed) VALUES (?, ?, ?, ?, ?, ?)'
                    prepared_stmt = session.prepare(insert_query)
                    batch = BatchStatement()
                    for result in results:
                        if isinstance(result, tuple) and len(result) == 6:
                            batch.add(prepared_stmt, result)
                        else:
                            print(f'Invalid result format: {result}')
                    try:
                        session.execute(batch)
                        st.success('Process completed! Click Next to proceed!')
                    except Exception as e:
                        st.error(f'Error executing batch insert: {e}')

                    # Save the place to session state
                    st.session_state['place'] = selected_destination
                    st.session_state['data_fetched'] = True
                    st.session_state.validity = True


elif st.session_state.page == 1:

    # Check if style.css exists
    style_path = 'style.css'
    if os.path.exists(style_path):
        with open(style_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Default styles
        st.markdown('''
        <style>
        .main {
            background-color: #000000;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f5;
        }
        </style>
        ''', unsafe_allow_html=True)
    place = st.session_state.place
    if not place:
        st.warning('Please go to the Home page and fetch the weather data first.')
    else:
        session = connect_to_cassandra()
        if session is None:
            st.error('Failed to connect to Cassandra. Exiting.')
        else:
            end_date = datetime.datetime.today().date()
            start_date = end_date - datetime.timedelta(days=365)
            weather_data = fetch_weather_data(session, place, start_date, end_date)
            print(weather_data)
            if weather_data.empty:
                st.error('No data found for the specified place and date range.')
            else:
                # Preprocess data
                X_resampled, y_resampled, label_encoder = preprocess_data(weather_data)

                # Train model
                model, scaler, accuracy, precision = train_model(X_resampled, y_resampled)

                # Make forecasts
                forecast_dates, forecast_conditions, temperature_data, humidity_data, windspeed_data = make_forecasts(
                    place, get_api_key('WEATHER API'))

                # Validate lengths
                if len(forecast_dates) == len(forecast_conditions) == len(temperature_data) == len(
                        humidity_data) == len(windspeed_data):
                    # Display forecasts

                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Weather Condition': forecast_conditions,
                        'Temperature': temperature_data,
                        'Humidity': humidity_data,
                        'Windspeed': windspeed_data
                    })

                    forecast_df['Image'] = forecast_df['Weather Condition'].apply(get_image_path)

                    # Display the images in a grid
                    st.markdown('### Weather condition prediction for ' + place + ' (One Week)')
                    cols = st.columns(len(forecast_df))

                    for idx, col in enumerate(cols):
                        image_path = forecast_df.iloc[idx]['Image']
                        # Ensure the image path is valid before displaying
                        if isinstance(image_path, str):
                            col.write(forecast_df['Date'][idx])
                            col.image(image_path, width=100)
                            col.write(forecast_df['Weather Condition'][idx])
                        else:
                            col.write('No image available')

                    # Calculate averages
                    avg_temp = sum(temperature_data) / len(temperature_data)
                    avg_humidity = sum(humidity_data) / len(humidity_data)
                    avg_windspeed = sum(windspeed_data) / len(windspeed_data)
                    st.markdown('### Metrics📊')
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Average Temperature🌡️', f'{avg_temp:.1f} °C')
                    col2.metric('Average Wind Speed💨', f'{avg_windspeed:.1f} mph')
                    col3.metric('Average Humidity💧', f'{avg_humidity:.1f}%')

                    st.markdown('### Temperature Line chart')
                    plost.line_chart(
                        data=forecast_df,
                        x='Date',
                        y='Temperature',
                        legend=None,
                        height=345,
                        use_container_width=True
                    )

                    c1, c2 = st.columns((3, 3))

                    with c1:
                        st.markdown('### Wind Speed Line Chart')
                        plost.line_chart(
                            data=forecast_df,
                            x='Date',
                            y='Windspeed',
                            legend=None,
                            height=345,
                            use_container_width=True
                        )

                    with c2:
                        st.markdown('### Humidity Line chart')
                        plost.line_chart(
                            data=forecast_df,
                            x='Date',
                            y='Humidity',
                            legend=None,
                            height=345,
                            use_container_width=True
                        )

                    # Decision on visiting
                    decision = should_visit(forecast_conditions)
                    st.subheader('Recommendation💡')
                    st.write(decision)
                    st.write('Prediction Accuracy🎯', f'{accuracy * 100.0:.2f}%')

                    # Save recommendation to session state
                    st.session_state['recommendation'] = decision
                else:
                    st.error('Data mismatch: Length of temperature, humidity, or windspeed data does not match the '
                             'length of forecast dates.')
                st.session_state.validity = True
elif st.session_state.page == 2:
    st.session_state.validity = True
    place = st.session_state.get('place', None)
    recommendation = st.session_state.get('recommendation', None)
    if not place or not recommendation:
        st.warning('Please complete the steps in the previous pages first.')
    else:
        # Fetch all locations
        session = connect_to_cassandra()
        rows = session.execute('SELECT codes, location FROM iata')
        destinations = [(row.codes, row.location) for row in rows]

        # Origin selection
        destination_list_origin = [f'{city}' for _, city in destinations]
        destination_list_origin.insert(0, 'None')
        col1, col2, col3, col4 = st.columns([19, 1, 6, 19])
        with col1:
            origin = st.selectbox('Origin🛫', destination_list_origin)
        with col3:
            st.image('/Users/mchalxz/Downloads/arrow-3.png')
        with col4:
            st.text_input('Destination🛬', place, disabled=True)
        if origin == 'None':
            st.warning('Please select an origin.')
            st.stop()
        col5, col6 = st.columns([1, 1])
        with col5:
            dates = st.date_input('Departure Date🗓️:')
        with col6:
            back_dates = st.date_input('Return Date🗓️:')
        if dates > back_dates:
            st.error('Return date cannot be earlier than departure date')
            st.stop()
        filter_list = ['None', 'Price Range', 'Only exact day', 'Cheapest']
        filters = st.selectbox('Filter', filter_list)
        if filters == 'Price Range':
            col9, col10 = st.columns(2)
            with col9:
                try:
                    min_price = int(st.text_input('Minimum price:'))
                except ValueError:
                    st.stop()
            with col10:
                try:
                    max_price = int(st.text_input('Maximum price:'))
                except ValueError:
                    st.stop()

        # Find the IATA code for the selected origin
        origin_iata = None
        for code, location in destinations:
            if location == origin:
                origin_iata = code
                break

        if not origin_iata:
            st.error(f'Could not find IATA code for the selected origin: {origin}')
        else:
            # Find the IATA code for the selected destination
            selected_iata_code = None
            for code, location in destinations:
                if location == place:
                    selected_iata_code = code
                    break

            if not selected_iata_code:
                st.error(f'Could not find IATA code for {place}.')
            else:
                st.write(f'Flight details from {origin} to {place} on ' + str(dates))
                st.session_state.temporigin = origin
                access_token = get_access_token(get_api_key('FLIGHT API'), get_api_key('SECRET API'))
                if access_token:
                    flight_offers = get_flight_offers(origin_iata, selected_iata_code, dates,
                                                      access_token)
                    flight_offers_limited = flight_offers.get('data', [])[:6]

                    if flight_offers_limited:
                        st.subheader('Flight Offers✈️:')
                        minprice = 10000000
                        for i in range(0, len(flight_offers_limited), 3):
                            cols = st.columns(3)
                            for idx, flight in enumerate(flight_offers_limited[i:i + 3]):
                                with cols[idx]:
                                    price = flight['price']['total']
                                    priceint = float(price.replace('RM', ''))
                                    if minprice > priceint:
                                        minprice = priceint
                                    segments = flight['itineraries'][0]['segments']
                                    plane_number = segments[0]['carrierCode'] + segments[0]['number']
                                    airline_company = segments[0]['carrierCode']
                                    origin = segments[0]['departure']['iataCode']
                                    destination = segments[-1]['arrival']['iataCode']
                                    departure_time = segments[0]['departure']['at']

                                    # Handle both formats for departure time
                                    try:
                                        # Attempt to parse with 'Z' at the end
                                        departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                        '%Y-%m-%dT%H:%M:%SZ')
                                    except ValueError:
                                        # Parse without 'Z'
                                        departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                        '%Y-%m-%dT%H:%M:%S')

                                    # Convert departure time to Malaysia time
                                    departure_time_malaysia = departure_time_utc.replace(
                                        tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kuala_Lumpur'))
                                    departure_time_malaysia_str = departure_time_malaysia.strftime(
                                        '%Y-%m-%d %H:%M:%S')

                                    date = departure_time_malaysia_str.split(' ')[0]
                                    time = departure_time_malaysia_str.split(' ')[1]
                                    if filters == 'Price Range':
                                        if min_price <= priceint <= max_price:
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    elif filters == 'None':
                                        background_color = 'background-color: ;'
                                    elif filters == 'Cheapest':
                                        if priceint == minprice:
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    else:
                                        if date == str(dates):
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    st.markdown(
                                        f'''
                                                                                            <div style='border: 1px solid #e1e1e1; border-radius: 10px; padding: 10px; margin-bottom: 10px; {background_color}'>
                                                                                                <h4>Flight {i + idx + 1}✈️:</h4>
                                                                                                <p><strong>Price:</strong> RM{price}</p>
                                                                                                <p><strong>Airline Company:</strong> {airline_company}</p>
                                                                                                <p><strong>Plane Number:</strong> {plane_number}</p>
                                                                                                <p><strong>Origin:</strong> {origin}</p>
                                                                                                <p><strong>Destination:</strong> {destination}</p>
                                                                                                <p><strong>Date (M'sia):</strong> {date}</p>
                                                                                                <p><strong>Time (M'sia):</strong> {time}</p>
                                                                                            </div>
                                                                                            ''', unsafe_allow_html=True
                                    )

                    else:
                        st.error('No flight offers available.')
                    st.write('')
                    st.write(f'Flight details from {place} to {st.session_state.temporigin} on ' + str(back_dates))
                    if access_token:
                        flight_offers = get_flight_offers(selected_iata_code, origin_iata, back_dates,
                                                          access_token)
                        flight_offers_limited = flight_offers.get('data', [])[:6]

                        if flight_offers_limited:
                            st.subheader('Flight Offers✈️:')
                            minprice = 10000000
                            for i in range(0, len(flight_offers_limited), 3):
                                cols = st.columns(3)
                                for idx, flight in enumerate(flight_offers_limited[i:i + 3]):
                                    with cols[idx]:
                                        price = flight['price']['total']
                                        priceint = float(price.replace('RM', ''))
                                        if minprice > priceint:
                                            minprice = priceint
                                        segments = flight['itineraries'][0]['segments']
                                        plane_number = segments[0]['carrierCode'] + segments[0]['number']
                                        airline_company = segments[0]['carrierCode']
                                        origin = segments[0]['departure']['iataCode']
                                        destination = segments[-1]['arrival']['iataCode']
                                        departure_time = segments[0]['departure']['at']

                                        # Handle both formats for departure time
                                        try:
                                            # Attempt to parse with 'Z' at the end
                                            departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                            '%Y-%m-%dT%H:%M:%SZ')
                                        except ValueError:
                                            # Parse without 'Z'
                                            departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                            '%Y-%m-%dT%H:%M:%S')

                                        # Convert departure time to Malaysia time
                                        departure_time_malaysia = departure_time_utc.replace(
                                            tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kuala_Lumpur'))
                                        departure_time_malaysia_str = departure_time_malaysia.strftime(
                                            '%Y-%m-%d %H:%M:%S')

                                        date = departure_time_malaysia_str.split(' ')[0]
                                        time = departure_time_malaysia_str.split(' ')[1]

                                        if filters == 'Price Range':
                                            if min_price <= priceint <= max_price:
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        elif filters == 'None':
                                            background_color = 'background-color: ;'
                                        elif filters == 'Cheapest':
                                            if priceint == minprice:
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        else:
                                            if date == str(back_dates):
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        st.markdown(
                                            f'''
                                                                                                                <div style='border: 1px solid #e1e1e1; border-radius: 10px; padding: 10px; margin-bottom: 10px; {background_color}'>
                                                                                                                    <h4>Flight {i + idx + 1}✈️:</h4>
                                                                                                                    <p><strong>Price:</strong> RM{price}</p>
                                                                                                                    <p><strong>Airline Company:</strong> {airline_company}</p>
                                                                                                                    <p><strong>Plane Number:</strong> {plane_number}</p>
                                                                                                                    <p><strong>Origin:</strong> {origin}</p>
                                                                                                                    <p><strong>Destination:</strong> {destination}</p>
                                                                                                                    <p><strong>Date (M'sia):</strong> {date}</p>
                                                                                                                    <p><strong>Time (M'sia):</strong> {time}</p>
                                                                                                                </div>
                                                                                                                ''',
                                            unsafe_allow_html=True
                                        )




                else:
                    st.error('Failed to obtain access token.')

col1, col2, col3 = st.columns([4, 5, 1])

with col1:
    if st.session_state.page > 0:
        st.button('Previous', on_click=prev_page)

with col3:
    if st.session_state.page < len(pages) - 1 and st.session_state.validity:
        st.button('Next', on_click=next_page)
