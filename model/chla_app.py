### import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from titlecase import titlecase
from sklearn.preprocessing import LabelEncoder


### set page configuration
st.set_page_config(page_icon="childrens-hospital-la-icon.jpg")


### set titles
col1, col2, col3 = st.columns([1,1,1])

with col2:
    st.image('childrens-hospital-la-logo.png')
    
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-family: Geneva; color: #005b98;'>Appointment No-Shows Prediction Tool</h3>", unsafe_allow_html=True)
st.caption("Welcome to the Appointment No-Show Prediction Model Web App for Children's Hospital Los Angeles (CHLA)! We understand that missed appointments can disrupt schedules and delay care for our young patients. That's why we've developed a cutting-edge predictive model to help identify the likelihood of appointment no-shows in advance. Our web app utilizes advanced machine learning algorithms trained on historical data to predict the probability of a patient missing their scheduled appointment.")


### ingest data
@st.cache_resource

def load_data(file_path):
    return pd.read_csv(file_path)


df = load_data("CHLA_clean_data_2024_Appointments.csv")
df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])

    
### date inputs
col1, col2 = st.columns([1,1])

with col1:
    start_datetime = st.date_input("Choose Start Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())
with col2:
    end_datetime = st.date_input("Choose End Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())

start_datetime = pd.to_datetime(start_datetime)
end_datetime = pd.to_datetime(end_datetime)

if start_datetime > end_datetime:
    st.error("End Date should be after Start Date")

    
### filter df by date inputs and return caption
if start_datetime and end_datetime:
    mask = (df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)
    filtered_df = df[mask]
    start_date = start_datetime.date()
    end_date = end_datetime.date()
    st.caption(f"You have selected appointments between {start_date} and {end_date}")
else:
    st.warning("Please select both start and end dates")   


### select and filter filtered_df by clinic
clinic_selector = st.multiselect("Select a Clinic", df['CLINIC'].unique())
if len(clinic_selector) == 0:  # Check if no clinic is selected
    filtered_df = filtered_df.copy()  # Retain the original DataFrame
else:
    filtered_df = filtered_df[filtered_df['CLINIC'].isin(clinic_selector)]
clinic_strings = []

for i in range(len(clinic_selector)):
    clinic_strings.append(titlecase(str(clinic_selector[i])))

clinic_string = ", ".join(clinic_strings)
st.caption(f"You have selected {clinic_string}")


### slice MRN
fdf = filtered_df[[
    'MRN',
    'APPT_DATE',
    'AGE',
    'CLINIC',
    'TOTAL_NUMBER_OF_CANCELLATIONS',
    'LEAD_TIME',
    'TOTAL_NUMBER_OF_RESCHEDULED',
    'TOTAL_NUMBER_OF_NOSHOW',
    'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT',
    'HOUR_OF_DAY',
    'NUM_OF_MONTH'
]]


### slice predictive df
pdf = fdf.drop(['MRN', 'APPT_DATE'], axis=1)


### load and run the predictor model
model = pickle.load(open('random_forest_model.pkl', 'rb'))


### label encoding
le = LabelEncoder()
object_cols = ['CLINIC']
for col in object_cols:
    pdf[col] = le.fit_transform(pdf[col])


### run model button
run_button = st.button('Run')

if run_button:
    
    try:
        ### run model and output predictions
        st.info("Your model is running...")    
        predictions = model.predict(pdf)
        predictions_series = pd.Series(predictions)
        fdf = fdf.reset_index(drop=True)
        final_df = pd.concat([fdf, predictions_series], axis=1)
        final_df.columns = [*final_df.columns[:-1], 'NO SHOW (Y/N)']
        final_df = final_df[['MRN', 'APPT_DATE', 'CLINIC', 'NO SHOW (Y/N)']]
        no_show_mapping = {0: 'NO', 1: 'YES'}
        final_df['NO SHOW (Y/N)'] = final_df['NO SHOW (Y/N)'].replace(no_show_mapping)
        final_df['MRN'] = final_df['MRN'].astype(str)
        final_df = final_df.sort_values(by='CLINIC')
        final_df = final_df.sort_values(by='APPT_DATE')
        final_df.rename(columns={'APPT_DATE': 'APPOINTMENT DATE'}, inplace=True)
        for index, row in final_df.iterrows():
            if row['NO SHOW (Y/N)'] == 'Yes':
                final_df.at[index, 'RECOMMENDATION'] = "DOUBLE-BOOK"
            else:
                final_df.at[index, 'RECOMMENDATION'] = "DON'T DOUBLE-BOOK" 
        st.success('Run Complete')   
        st.write(final_df)


        ### dashboard
        c1, c2 = st.columns(2)
        c1.metric(label='No Shows',
                   value=len(final_df[final_df['NO SHOW (Y/N)']=='YES']))
        c2.metric(label='Shows',
                   value=len(final_df[final_df['NO SHOW (Y/N)']!='YES']))


        ### download report button
        csv_string = final_df.to_csv(index=False)  # convert to predcitions df to csv
        export_report_button = st.download_button("Download Report",
                                                  csv_string,
                                                  file_name="final_report.csv",
                                                  mime="text/csv")
    except Exception as e:
        st.error("Error: Choose Correct Inputs")

    
### links
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("[🏥 CHLA](https://www.chla.org/)")
st.markdown("[🐱 GitHub](https://github.com/svanhemert00/chla-no-show-web-app)")
