# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Helpful resources:
# * [Guide to Markdown](https://paperhive.org/help/markdown)
# * [Guide to LaTeX Math symbols](http://tug.ctan.org/info/undergradmath/undergradmath.pdf)
# * [Python Cheat Sheets](https://ehmatthes.github.io/pcc/cheatsheets/README.html)
# * [Moving to Python from MATLAB](https://bastibe.de/2013-01-20-a-python-primer-for-matlab-users.html)
#
# ### Template for lab report:
# * **Title:** Name the lab that this report is for
# * **Collaborators:** Team work on labs is encouraged, but everyone is required to turn in their own lab report. Please list your collaborators in the intro to the report.
# * **Goal and Introduction:** Add a brief description of the goal and background knowledge for the lab. This can be drawn from the lab description, but should be in your own words.
# * **Data:** List the datasets used, what they describe and any quality/pre-processing before analysis.
# * **Approach and Results:** Describe your approach for each question in the lab description and interpretation of the results for that question.
# * **Conclusions:** Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# * **References:** Cite any resources or publications you used.
# ---
# # Final Project
# ### Elise Segal, Daniel Heilmen, Natalie Giovi, Percy Slattery
#
# ## Goal and Introduction
# Add a brief description of the goal and background knowledge for the lab. This can be drawn from the lab description, but should be in your own words.
#
# ## Data
# List the datasets used, what they describe and any quality/pre-processing before analysis.
#
# ----
# ## Approach and Results
# Describe your approach for each question in the lab description and interpretation of the results for that question.
# Start with an over-arching paragraph to describe your approach as you see fit.

# %%
# imports the libraries needed for the project
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# time converter to datetime object for the OMNI data
tconvert = lambda x: dt.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S.%fZ')
# reads the OMNI data into arrays
data = np.genfromtxt('OMNI20032024.csv', names=True, delimiter=',', skip_header=94, encoding='utf-8',converters={0:tconvert}, dtype=None)

time = data['TIME_AT_CENTER_OF_HOUR_yyyymmddThhmmsssssZ']
swavgB = np.array(data['1AU_IP_MAG_AVG_B_nT'], dtype = float)
swvelocity = np.array(data['1AU_IP_PLASMA_SPEED_Kms'], dtype = float)
swpressure = np.array(data['1AU_IP_FLOW_PRESSURE_nPa'], dtype = float)
swtemp = np.array(data ['1AU_IP_PLASMA_TEMP_Deg_K'], dtype = float)
swdensity = np.array(data['1AU_IP_N_ION_Per_cc'], dtype = float)
dst = np.array(data['1H_DST_nT'], dtype = float)

# %%
# values that need to be filtered due to null data points
arrays = [swavgB, swdensity, swvelocity, swpressure, swtemp]
# maxs of the legitimate from OMNI
filters = [80, 100, 1200, 60, 1*10**7]
# filters out non data points to be null
for array, threshold in zip(arrays, filters):
    for index, value in enumerate(array):
        if value >= threshold:
            array[index] = np.nan

# %%
# correlates data to labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure),
        'Plasma Temperature|Temperature (K)': (time,swtemp),
        'Ion Number Density|Density (per cc)': (time,swdensity),
        'DST Index|DST (nT)': (time,dst)}
fig, axes = plt.subplots(6,1, figsize = (12,20))
# for loop to add data to each plot
for ax, (label, (x, y)) in zip(axes.flat, data.items()):
    #add data to the plot
    ax.plot(x,y)
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel(r'Date $(year)$')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %% [markdown]
# Between 2019 and 2022 most of the data is consistent, showing low solar activity. The end of the data specifically around 2024 shows a lot of variety demostrating higher solar activy aligning with the solar cycle with solar min in 2019 and solar max in 2024.

# %% [markdown]
# While the most notable solar event from the plots is the May 2024 storm it is not the best example to show how solar wind parameters behave during a CME. This is because the May 2024 storm is the result of a cannibal CME which means that there were several CMEs that were back to back only a few hours apart. Therefore we decided to go with the April 23rd storm which was the result on a single CME to showcase how solar wind changes during a solar event. 

# %%
# creates a new figure for plots
fig, axes = plt.subplots(3,2, figsize = (12,12))
fig.suptitle('April 2023 CME')
# for loop to add data to each plot
for ax, (label, (x, y)) in zip(axes.flat, data.items()):
    # add data to the plot
    # narrows graph to just time around the 4/23/2023 CME
    ax.set_xlim(dt.datetime(2023,4,22),dt.datetime(2023,4,27))
    ax.plot(x,y)
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel('Date')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %% [markdown]
# During a CME the magnetic field at 1 AU increases at the arrival of the CME and conitnues to increase before decreasing after the passing of the CME. The solar wind velocity also increases when the storm arrives before decreasing slightly again and remaining steady. There is a small peak in the solar wind pressure, density and temperature at the arrival of the CME before it decreases again. Lastly, the DST Index decreases to negative during the CME before recovering. This behavior matches the the structure of a CME. Typically a CME will have a shock associated with where the fast solar wind from the CME over takes the slow solar wind in front forming the shock. However, there does not have to be a shock associated with the CME. This causes an increase in solar wind speed when the shock and CME reach Lagrange point 1. Since the shock is at the beginning there is a bigger increase at the start of the CME before decreasing a little while the magnetic cloud of the CME passes. In the magnetic cloud of a CME there is a magnetic field as the name suggests. However the magnetic cloud mainly contains the magnetic cloud and nothing else. Therefore, the magnetic field should increase at the arrival of the shock and continue to increase and stay at a higher value during the passing of the magnetic cloud. However the solar wind parameter should increase at the arrival of the shock due to the compression of the solar wind but then drop during the passing of the magnetic cloud which is seen with the April 23rd storm of 2023. Lastly the DST measures the impact of the magnetic field on Earth surface with Earth's magnetic field therefore is showcases the opposite effect of Lagrange point 1 by decreasing due to the CME before recovering back to around 0 nT.

# %%
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# Helper function: interpolate missing (NaN) values in the time series using linear interpolation
def interpolate_nan(y):
    nans = np.isnan(y)                          # Identify NaN positions
    not_nans = ~nans                            # Identify valid (non-NaN) positions
    x = np.arange(len(y))                       # Create an index array
    y[nans] = np.interp(x[nans], x[not_nans], y[not_nans])  # Interpolate NaNs
    return y

def analyze_fft(data, dt_hours=1, top_k=6):
    # Interpolate NaNs and prepare data
    y = interpolate_nan(data.copy())
    dt_days = dt_hours / 24
    n = len(y)

    # Detrend and FFT
    y_detrended = y - np.mean(y)
    Y = fft(y_detrended)
    freq = fftfreq(n, d=dt_days)
    power = np.abs(Y)**2

    # Filter to only positive frequencies
    pos_mask = freq > 0
    pos_freq = freq[pos_mask]
    pos_power = power[pos_mask]
    pos_indices = np.where(pos_mask)[0]

    # Get top-k harmonics
    harmonics = []
    for _ in range(top_k):
        idx_in_pos = np.argmax(pos_power)
        global_idx = pos_indices[idx_in_pos]
        harmonic = {
            'amplitude': power[global_idx],
            'frequency': freq[global_idx],
            'period_days': 1 / freq[global_idx],
            'harmonic_index': global_idx
        }
        harmonics.append(harmonic)
        pos_power[idx_in_pos] = 0  # Remove this peak

    return freq, power, n, harmonics

def sort_harmonics_amp(harmonics):
    sorted_h = sorted(harmonics, key=lambda h: h['amplitude'], reverse=True)
    return sorted_h


# %%
#correlates data with labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure),
        'Plasma Temperature|Temperature (K)': (time,swtemp),
        'Ion Number Density|Density (per cc)': (time,swdensity),
        'DST Index|DST (nT)': (time,dst)}
fig, axes = plt.subplots(3,2, figsize = (12,12))
# for loop to add data to each plot
for ax, (label, (x, i)) in zip(axes.flat, data.items()):
    freq, power, n, harmonics = analyze_fft(i)
    sorted_amps = sort_harmonics_amp(harmonics)
    #add data to the plot
    ax.plot(freq[:n//2], power[:n//2])
    # adds proper titles and labels
    title, space, unit = label.partition('|')
    ax.set_title(f'Power Spectrum of {title}')   
    ax.set_xlabel("Frequency (cycles per day)")
    ax.set_ylabel(f"Power ({unit}^2)")
    ax.set_xlim(-0.01,2)

    freq = abs(freq)

    dominant_frequencies = []

    # Loop to find the top 5 dominant frequencies
    for i in range(5):
        # Find the index of the dominant frequency (maximum power) within the valid range
        dominant_idx = np.argmax(power)
        # Extract the dominant frequency in cycles/day
        dominant_freq = freq[dominant_idx]
        # add to array
        dominant_frequencies.append(dominant_freq)
        # delete max power to find second dominant frequency
        power = np.delete(power, dominant_idx)  

    # Print the results
    print(f'{title}:')
    for rank, freq in enumerate(dominant_frequencies, start=1):
        days = 1 / freq
        years = days / 365
        print(f'\t {rank} Dominant frequency: {freq:.4f} cycles per day (~{days:.2f} days or ~{years:.4f} years)')


fig.tight_layout()

# %%
from matplotlib.dates import num2date, date2num
# correlates data to labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB, 10),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity,50),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure,10),
        'Plasma Temperature|Temperature (K)': (time,swtemp,0.5),
        'Ion Number Density|Density (per cc)': (time,swdensity,10),
        'DST Index|DST (nT)': (time,dst,20)}
fig, axes = plt.subplots(3,2, figsize = (12,20))
# for loop to add data to each plot
for ax, (label, (x, y, amp)) in zip(axes.flat, data.items()):
    #add data to the plot
    t = np.arange(0, x.size,1)
    ax.plot(x,y)
    ax.plot(x, amp*np.cos((0.0002/24)*t)+np.mean(y))
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel(r'Date $(year)$')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %% [markdown]
# ### Question 1
# Write a function to read the *.csv files using numpy.genfromtxt. Leverage the example above to ensure success.
#


# %% [markdown]
# ### Question 3
# Description of what you need to do and interpretation of results (if applicable)


# %% [markdown]
# ## Conclusions
# Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# ## References
# List any references used
