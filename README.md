# P03---Docker-Streamlit
Welcome to [chlanoshow.streamlit.app](https://chlanoshow.streamlit.app/) web application!

This predictive tool is tailored for professionals affiliated with the Children's Hospital of Los Angeles (CHLA), including physicians, nurses, clerical staff, and administrators. Its primary objective is to enable informed decision-making by forecasting patient attendance at appointments based on provided data. Such insights can facilitate strategic scheduling adjustments, including the consideration of double booking scenarios, among other operational considerations.

## Files:
- [chla_app.py](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/chla_app.py): file that generates the user interface, takes in user inputs, runs the model, and returns predicted results
- [project_02.ipynb](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/project_02.ipynb): Jupyter Notebook for initial model exploration and building.
- [tests.ipynb](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/tests.ipynb): Jupyter Notebook to test data transformations found in [chla_app.py](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/chla_app.py)
- Data Files:
  - [CHLA_clean_data_2024_Appointments.csv](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/CHLA_clean_data_2024_Appointments.csv): test data
  - [CHLA_clean_data_until_2023.csv](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/CHLA_clean_data_until_2023.csv): training data
- Models:
  - [random_forest_model.pkl](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/random_forest_model.pkl): serialized Random Forest
  - [label_encoder.pkl](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/label_encoder.pkl): serialized label encoder 
- Image Files:
  - [childrens-hospital-la-icon.jpg](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/childrens-hospital-la-icon.jpg): favicon
  - [childrens-hospital-la-logo.png](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/childrens-hospital-la-logo.png): main logo
- [requirements.txt](https://github.com/svanhemert00/chla-no-show-web-app/blob/main/requirements.txt): contains the library dependencies required to build the web app
