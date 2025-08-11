# Avocado Price Analysis

A time series analysis and forecasting project for avocado prices using Facebook Prophet.

## Description

This project analyzes historical avocado price data and creates forecasting models to predict future price trends. The analysis includes data exploration, visualization, and time series forecasting using the Prophet library.

## Requirements

```
pandas
numpy
matplotlib
seaborn
prophet
```



## Data

The project expects an avocado dataset in CSV format located at `data/avocado.csv`. The dataset should contain:
- Date column
- AveragePrice column
- region column
- year column

## Usage

Run the analysis script to:

1. Load and explore the avocado price data
2. Create visualizations showing price trends over time
3. Analyze data distribution by region and year
4. Build Prophet forecasting models for overall data and specific regions
5. Generate forecasts for the next 365 days

## Analysis Components

### Data Exploration
- Basic data inspection and sorting by date
- Price trend visualization over time
- Regional and yearly data distribution analysis

### Forecasting Models

#### Global Model
- Uses all available data
- Forecasts 365 days into the future
- Includes trend and seasonal components

#### Regional Model (West Region)
- Focuses on West region data specifically
- Provides region-specific forecasting
- Includes component analysis showing trends and seasonality

## Output

The script generates:
- Line plots showing historical price trends
- Count plots for regional and yearly data distribution
- Prophet forecast plots with confidence intervals
- Component plots showing trend, weekly, and yearly seasonality

## Files

- Main analysis script (Jupyter notebook format with #%% cell separators)
- `data/avocado.csv` - Input dataset

## Notes

- The script includes two separate analyses: one using all data and another focusing on the West region
- Prophet automatically handles missing data and outliers
- Forecasts include uncertainty intervals
- Component plots help understand underlying patterns in the data