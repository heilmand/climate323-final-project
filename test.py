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
swavgB = data['1AU_IP_MAG_AVG_B_nT']
swvelocity = data['1AU_IP_PLASMA_SPEED_Kms']
swpressure = data['1AU_IP_FLOW_PRESSURE_nPa']
swtemp = data ['1AU_IP_PLASMA_TEMP_Deg_K']
swdensity = data['1AU_IP_N_ION_Per_cc']
dst = data['1H_DST_nT']

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

# Copy the solar wind velocity data (assumed to be defined earlier)
y = swvelocity.copy()

# Fill in missing values using linear interpolation
y = interpolate_nan(y)

# Define the time step in days (data assumed to be hourly)
dt_hours = 1            # Time step in hours
dt_days = dt_hours / 24 # Convert to days
n = len(y)              # Number of observations

# Remove the mean from the data to focus on fluctuations (detrending)
y_detrended = y - np.mean(y)

# Perform Fast Fourier Transform (FFT) to move to frequency domain
Y = fft(y_detrended)

# Get the frequency values corresponding to FFT result (in cycles per day)
freq = fftfreq(n, d=dt_days)

# Compute the power spectrum (squared magnitude of FFT result)
power = np.abs(Y)**2

# Plot the power spectrum (positive frequencies only)
plt.figure(figsize=(10, 4))
plt.plot(freq[:n//2], power[:n//2])            # Plot only positive frequencies
plt.title("Power Spectrum of Solar Wind Velocity")
plt.xlabel("Frequency (cycles per day)")
plt.ylabel("Power")
plt.xlim(0, 0.4)                                # Focus on 0–0.4 cycles/day (~2.5 days and longer)
plt.grid()
plt.show()

# Define frequency range of interest: 0.01 to 0.2 cycles/day (~5 to 100 days)
#valid_range = (freq > 0.01) & (freq < 0.2)

# Find the index of the dominant frequency (maximum power) within the valid range
dominant_idx = np.argmax(power)
# Extract the dominant frequency in cycles/day
fundamental_freq = freq[dominant_idx]

# Print the dominant frequency and corresponding period (in days)
print(f"Fundamental frequency: {fundamental_freq:.4f} cycles per day (~{1/fundamental_freq:.1f} days)")
print(swvelocity.size/24)


amp1 = np.argmax(power) +1
#shows the amplitude and frequency and period value for the largest amplitude
print(f'Largest amplitude {power[np.argmax(power)]} and corresponding frequency {freq[np.argmax(power)]}')
print(f'and period {1/freq[np.argmax(power)]}')
print(f'{amp1}th harmonic is associated with the largest amplitude') 
# prints harmonic associated with largest amplitude
# finds the second amplitude with argmax and stores the corresponding harmonic and stores it in amp2
# plus 1 to account for index starting at 0 for arrays
# removes amp1 with delete from the dataset to find the second largest
amp2 = np.argmax(np.delete(power, amp1-1)) + 1
#shows the amplitude and frequency and period value for the second largest amplitude
print(f'Second largest amplitude {power[np.argmax(np.delete(power, amp1-1))]}') 
print(f'and corresponding frequency {freq[np.argmax(np.delete(power, amp1-1))]}')
print(f'and period {1/np.abs(freq[np.argmax(np.delete(power, amp1-1))])/365}')
print(f'{amp2}st harmonic is associated with the second largest amplitude') 
# prints harmonic associated with second largest amplitude

# %% [markdown]
# ### Question 1
# Write a function to read the *.csv files using numpy.genfromtxt. Leverage the example above to ensure success.
#
# Here is the ftt
# from scipy.fftpack import fft, ifft, fftfreq
# from scipy.signal import butter, filtfilt
# import numpy as np
# import matplotlib.pyplot as plt
#
# Helper function: interpolate missing (NaN) values in the time series using linear interpolation
# def interpolate_nan(y):
#     nans = np.isnan(y)                          # Identify NaN positions
#     not_nans = ~nans                            # Identify valid (non-NaN) positions
#     x = np.arange(len(y))                       # Create an index array
#     y[nans] = np.interp(x[nans], x[not_nans], y[not_nans])  # Interpolate NaNs
#     return y
#
# Copy the solar wind velocity data (assumed to be defined earlier)
# y = swvelocity.copy()
#
# Fill in missing values using linear interpolation
# y = interpolate_nan(y)
#
# Define the time step in days (data assumed to be hourly)
# dt_hours = 1            # Time step in hours
# dt_days = dt_hours / 24 # Convert to days
# n = len(y)              # Number of observations
#
# Remove the mean from the data to focus on fluctuations (detrending)
# y_detrended = y - np.mean(y)
#
# Perform Fast Fourier Transform (FFT) to move to frequency domain
# Y = fft(y_detrended)
#
# Get the frequency values corresponding to FFT result (in cycles per day)
# freq = fftfreq(n, d=dt_days)
#
# Compute the power spectrum (squared magnitude of FFT result)
# power = np.abs(Y)**2
#
# Plot the power spectrum (positive frequencies only)
# plt.figure(figsize=(10, 4))
# plt.plot(freq[:n//2], power[:n//2])            # Plot only positive frequencies
# plt.title("Power Spectrum of Solar Wind Velocity")
# plt.xlabel("Frequency (cycles per day)")
# plt.ylabel("Power")
# plt.xlim(0, 0.4)                                # Focus on 0–0.4 cycles/day (~2.5 days and longer)
# plt.grid()
# plt.show()
#
# Define frequency range of interest: 0.01 to 0.2 cycles/day (~5 to 100 days)
# valid_range = (freq > 0.01) & (freq < 0.2)
#
# Find the index of the dominant frequency (maximum power) within the valid range
# dominant_idx = np.argmax(power)
#
# Extract the dominant frequency in cycles/day
# dominant_freq = freq[dominant_idx]
#
# Print the dominant frequency value (for debugging or info)
# print(dominant_freq)
#
# Print the dominant frequency and corresponding period (in days)
# print(f"Dominant frequency: {dominant_freq:.4f} cycles per day (~{1/dominant_freq:.1f} days)")
# %% [markdown]
# ### Question 2
# Description of what you need to do and interpretation of results (if applicable)
#
# here is the removal of periodicities
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, sosfiltfilt
# import datetime as dt
#
# Function to apply a bandstop filter 
# def bandstop_filter_sos(data, lowcut, highcut, fs, order=4):
#     nyq = 0.5 * fs  # Nyquist frequency (half the sampling rate)
#     low = lowcut / nyq  # Normalize lower bound
#     high = highcut / nyq  # Normalize upper bound
#     # Create a bandstop Butterworth filter (does not distort the other signals when filtering)
#     sos = butter(order, [low, high], btype='bandstop', output='sos')
#     # Apply the filter with zero-phase distortion (preserves the timing of events)
#     return sosfiltfilt(sos, data) 
#
# Sampling frequency 
# fs = 1 / dt_days  #Sampling frequency in samples per day (e.g., 24 for hourly data)
#
# Filter out the 27-day periodic signal (common in solar wind data)
# lowcut1 = 0.034   # Lower edge of the 27-day band in cycles/day (~29.4 days)
# highcut1 = 0.040  # Upper edge (~25 days)
# Apply bandstop filter to remove the 27-day signal
# y_filtered_27 = bandstop_filter_sos(y, lowcut1, highcut1, fs)
#
# Filter out a second dominant frequency found from FFT
# bandwidth = 0.003  # Width of the band to remove (you can adjust this)
# lowcut2 = dominant_freq2 - bandwidth  # Lower bound around the second frequency
# highcut2 = dominant_freq2 + bandwidth  # Upper bound
# Apply bandstop filter again to remove the second dominant frequency
# y_filtered_27_and_2 = bandstop_filter_sos(y_filtered_27, lowcut2, highcut2, fs)
#
# Plot the original and filtered signals for visual comparison
# plt.figure(figsize=(12, 5))
# plt.plot(time, y, label='Original', linewidth=1)  # Original signal
# plt.plot(time, y_filtered_27, label='Filtered (27-day removed)', linewidth=1, alpha=0.8)
# plt.plot(time, y_filtered_27_and_2, label='Filtered (27-day + second freq removed)', linewidth=1, alpha=0.8)
# plt.title('Solar Wind Velocity with Dominant Periodicities Removed')
# plt.xlabel('Time')
# plt.ylabel('Velocity (km/s)')
# plt.legend()
#
# focus on a specific date range
# plt.xlim(dt.datetime(2023, 1, 1), dt.datetime(2023, 6, 1))
# plt.grid()
# plt.show()
#
# Debugging info to check for data integrity 
# print("NaNs in final signal:", np.isnan(y_filtered_27_and_2).sum(), "Infs:", np.isinf(y_filtered_27_and_2).sum())
# print("NaNs in original signal:", np.isnan(y).sum()) 
# print("dt_days =", dt_days)  #Print the time step to verify consistency
#
# Print the normalized filter bounds for verification
# nyq = 0.5 * fs  #Recalculate Nyquist frequency for clarity
# low1 = lowcut1 / nyq
# high1 = highcut1 / nyq
# low2 = lowcut2 / nyq
# high2 = highcut2 / nyq
# print(f"27-day bandstop -> low = {low1:.4f}, high = {high1:.4f}")
# print(f"Second bandstop -> low = {low2:.4f}, high = {high2:.4f}")
# %% [markdown]
# ### Question 3
# Description of what you need to do and interpretation of results (if applicable)
# here is the comparison
# FFT of original signal
# n = len(y)
# Y_orig = fft(y - np.mean(y))  #Detrended original signal
# freq = fftfreq(n, d=dt_days)
# power_orig = np.abs(Y_orig)**2
#
# FFT of filtered signal
# Y_filtered = fft(y_filtered_27_and_2 - np.mean(y_filtered_27_and_2))
# power_filtered = np.abs(Y_filtered)**2
#
# Plot power spectra before and after filtering
# plt.figure(figsize=(12, 5))
# plt.plot(freq[:n//2], power_orig[:n//2], label='Original', color='gray', alpha=0.6)
# plt.plot(freq[:n//2], power_filtered[:n//2], label='Filtered', color='blue')
# plt.title("Power Spectrum Before and After Filtering")
# plt.xlabel("Frequency (cycles per day)")
# plt.ylabel("Power")
# plt.legend()
# plt.xlim(0, 0.1)  #Zoom into meaningful range
# plt.grid()
# plt.show()
#
# print("Total power before filtering:", np.sum(power_orig))
# print("Total power after filtering:", np.sum(power_filtered))
# %% [markdown]
# ## Conclusions
# Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# ## References
# List any references used
