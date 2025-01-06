# Medical Premium Calculator

A Streamlit web application for calculating and comparing medical insurance premiums.

## Features

- Compare two insurance plans side by side
- Calculate premiums with inflation rates
- View premium projections and 5-year averages
- Support for both HKD and USD currencies
- Interactive graphs and tables

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/medical-premium-calculator.git
cd medical-premium-calculator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Format

The application expects an Excel file named `premiumtable.xlsx` with two sheets:
- 'female': Premium data for female customers
- 'male': Premium data for male customers

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- Openpyxl
