# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.set_page_config(
        page_title="Forecast de Acciones",
        page_icon="chart_with_upwards_trend"
    )

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Predicción de Precios de Acciones')



try : 
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.text_input("Introuce el ticker de la acción")
                    # st.selectbox('Select dataset for prediction', stocks)

    years = (1,2,3,4)
    n_years = st.selectbox('Años a predecir', years)
    period = n_years * 365


    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')  # Elimina el timezone y formatea la fecha

        return data

    data_load_state = st.text('')
    data = load_data(selected_stock)
    data_load_state.text('Datos cargados!')
    data_load_state.text('')

    st.subheader('Datos históricos')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Precio de apertura"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Precio de cierre"))
        fig.layout.update(title_text='Serie Temporal', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()



    ##################
    # Forecasting with Facebook Prophet

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns = {'Date': 'ds','Close':'y'})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    descargar_forecast = forecast.to_csv()

    st.subheader("Datos del Forecast")
    st.write(forecast.tail())
    st.download_button(
    label="Descargar",
    data=descargar_forecast,
    file_name='forecast.csv',
    mime='text/csv'
)

    st.write("")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

except ValueError:
    st.success("Introduce un ticker para comenzar.")