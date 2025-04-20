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

# %% [markdown]
# ## Data Collection
# Start by importing the data from OMNI. We used a csv file and data from 2000 to 2024 to show 2 solar cycles. From OMNI we chose hourly data because trying to get minute data from OMNI takes too long to upload into a notebook. From OMNI, we retrieved time of the measurement, the average magnetic field at 1 AU, the solar wind speed at 1 AU, the solar wind pressure at 1 AU, the solar wind temperature at 1 AU, the solar wind density at 1 AU and the DST Index from stations around the equator on the surface of the Earth.

# %%
# imports the libraries needed for the project
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# time converter to datetime object for the OMNI data
tconvert = lambda x: dt.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S.%fZ')
# reads the OMNI data into arrays
data = np.genfromtxt('OMNI2_20002024.csv', names=True, delimiter=',', skip_header=97, encoding='utf-8',converters={0:tconvert}, dtype=None)

time = data['TIME_AT_CENTER_OF_HOUR_yyyymmddThhmmsssssZ']
swavgB = np.array(data['1AU_IP_MAG_AVG_B_nT'], dtype = float)
swvelocity = np.array(data['1AU_IP_PLASMA_SPEED_Kms'], dtype = float)
swpressure = np.array(data['1AU_IP_FLOW_PRESSURE_nPa'], dtype = float)
swtemp = np.array(data ['1AU_IP_PLASMA_TEMP_Deg_K'], dtype = float)
swdensity = np.array(data['1AU_IP_N_ION_Per_cc'], dtype = float)
dst = np.array(data['1H_DST_nT'], dtype = float)

# %% [markdown]
# Due to OMNI being real world data there are some times where there was no measurements and OMNI records these not as Nans but as the highest order minus 1. Therefore, to prevent these numbers from being read, we filtered them out using different thresholds of the max value each of the types of parameters can have.

# %%
# values that need to be filtered due to null data points
arrays = [swavgB, swdensity, swvelocity, swpressure, swtemp]
# maxs of the legitimate from OMNI
filters = [99, 999, 9999, 99, 9999999]
# filters out non data points to be null
for array, threshold in zip(arrays, filters):
    for index, value in enumerate(array):
        if value >= threshold:
            array[index] = np.nan

# %% [markdown]
# First step is to visualize the data to ensure some periodic behavior exists for the fft.

# %%
# correlates data to labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure),
        'Plasma Temperature|Temperature (K)': (time,swtemp),
        'Ion Number Density|Density (per cc)': (time,swdensity),
        'DST Index|DST (nT)': (time,dst)}
fig, axes = plt.subplots(2,3, figsize = (15,8))
# for loop to add data to each plot
for ax, (label, (x, y)) in zip(axes.flat, data.items()):
    #add data to the plot
    ax.plot(x,y, color = '#bd1caa')
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel(r'Date $(year)$')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %% [markdown]
# The plots show the solar cycle with solar maximums around 2003, 2014 and 2024 and the solar minimums around 2008 and 2019. Overall, all the solar wind parameters show increases during solar maximums and lower values in solar minimum years. The DST being negative acts in the opposite way, decreasing during solar maximum. This suggests there is periodic behavior and we can then proceed with the fft.
#

# %% [markdown]
# While the most notable solar events from the plots during solar maximums, it is not the best example to show how solar wind parameters behave during a CME. This is because the they are major storms that could have been the result of multiple CMEs back to back (cannibal CME). Therefore we decided to go with the April 23rd, 2023 storm which was the result on a single CME to showcase how solar wind changes during a solar event.

# %%
# creates a new figure for plots
fig, axes = plt.subplots(2,3, figsize = (15,8))
fig.suptitle('April 2023 CME')
# for loop to add data to each plot
for ax, (label, (x, y)) in zip(axes.flat, data.items()):
    # add data to the plot
    # narrows graph to just time around the 4/23/2023 CME
    ax.set_xlim(dt.datetime(2023,4,22),dt.datetime(2023,4,27))
    ax.plot(x,y, color = '#bd1caa')
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel('Date')
    ax.set_ylabel(ytext)
    ax.tick_params(axis='x', rotation=45)
fig.tight_layout()


# %% [markdown]
# During a CME the magnetic field at 1 AU increases at the arrival of the CME and conitnues to increase before decreasing after the passing of the CME. The solar wind velocity also increases when the storm arrives before decreasing slightly again and remaining steady. There is a small peak in the solar wind pressure, density and temperature at the arrival of the CME before it decreases again. Lastly, the DST Index decreases to negative during the CME before recovering. This behavior matches the the structure of a CME. Typically a CME will have a shock associated with where the fast solar wind from the CME over takes the slow solar wind in front forming the shock. However, there does not have to be a shock associated with the CME. This causes an increase in solar wind speed when the shock and CME reach Lagrange point 1. Since the shock is at the beginning there is a bigger increase at the start of the CME before decreasing a little while the magnetic cloud of the CME passes. In the magnetic cloud of a CME there is a magnetic field as the name suggests. However the magnetic cloud mainly contains the magnetic cloud and nothing else. Therefore, the magnetic field should increase at the arrival of the shock and continue to increase and stay at a higher value during the passing of the magnetic cloud. However the solar wind parameter should increase at the arrival of the shock due to the compression of the solar wind but then drop during the passing of the magnetic cloud which is seen with the April 23rd storm of 2023. Lastly the DST measures the impact of the magnetic field on Earth surface with Earth's magnetic field therefore is showcases the opposite effect of Lagrange point 1 by decreasing due to the CME before recovering back to around 0 nT.

# %% [markdown]
# Next step is to do the fft however the fft cannot have any NaN values. To start we created an interpolate function that replaces any NaN values with values similar that are around it allowing us to proceed with the fft and ifft later. Then we created a function that takes the fft and returns arrays allowing us to plot the power spectrum.

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
    
    #gets ftt
    data_fft = fft(y)/y.size # Normalize it
    freqs = fftfreq(y.size, 1/24) # hours per day

    power = np.abs(data_fft[1:y.size//2]**2) # Get that power
    freq = freqs[1:y.size//2] # This is cycles-per-day.

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


# %% [markdown]
# Plots the power spectrums for each parameter and finds the top 5 dominant frequencies in each parameter. The dominant frequencies are found by finding the max power and then the corresponding frequency and adding it to an array then deleting that value and finding the new max.
#

# %%
#correlates data with labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure),
        'Plasma Temperature|Temperature (K)': (time,swtemp),
        'Ion Number Density|Density (per cc)': (time,swdensity),
        'DST Index|DST (nT)': (time,dst)}

fig, axes = plt.subplots(2,3, figsize = (15,8))
# for loop to add data to each plot
fig.suptitle('Power Spectrums')

for ax, (label, (x, i)) in zip(axes.flat, data.items()):
    freq, power, n, harmonics = analyze_fft(i)
    sorted_amps = sort_harmonics_amp(harmonics)
    #add data to the plot
    ax.plot(freq[:n//2], power[:n//2], color = '#bd1caa')
    # adds proper titles and labels
    title, space, unit = label.partition('|')
    ax.set_title(f'Power Spectrum of {title}')   
    ax.set_xlabel("Frequency (cycles per day)")
    ax.set_ylabel(f"Power ({unit}^2)")
    ax.set_xlim(-0.01,1.5)

fig.tight_layout()

# %%
#arrays for storing the dominant frequencies
dom_amps = []
dom_freqs = []

for ax, (label, (x, i)) in zip(axes.flat, data.items()):
    freq, power, n, harmonics = analyze_fft(i)
    sorted_amps = sort_harmonics_amp(harmonics)
    freq = abs(freq)

    dominant_frequencies = []

    # Loop to find the top 5 dominant frequencies
    for i in range(5):
        # Find the index of the dominant frequency (maximum power) within the valid range
        dominant_idx = np.argmax(power)
        dom_amps.append(np.sqrt(power[dominant_idx]))
        # Extract the dominant frequency in cycles/day
        dominant_freq = freq[dominant_idx]
        dom_freqs.append(dominant_freq)
        # add to array
        dominant_frequencies.append(dominant_freq)
        # delete max power to find second dominant frequency
        power = np.delete(power, dominant_idx)  

    # Print the results
    print(f'\n{title}:')
    for rank, freq in enumerate(dominant_frequencies, start=1):
        days = 1 / freq
        years = days / 365
        print(f'\t {rank} Dominant frequency: {freq:.4f} cycles per day (~{days:.2f} days or ~{years:.4f} years)')

# %% [markdown]
# Dominant frequencies that are repeated through multiple of the parameters are 12 years, 27 days and 9 days. The 12 years correlates to the solar cycle and the 27 days correlates to the solar rotation cycle so these as dominant frequencies is expected.
#

# %% [markdown]
# Removes the dominant frequencies using a band pass filter and ifft. Removes all frequencies around 5 days of the 2 dominant frequencies in days and 3 months for the dominant frequency in years.
#

# %%
# original data arrays
arrays = [swavgB, swdensity, swvelocity, swpressure, swtemp, dst]
# new arrays for the filtered data
swavgB_filt = np.empty(swavgB.size)
swdensity_filt = np.empty(swdensity.size)
swvelocity_filt = np.empty(swvelocity.size)
swpressure_filt = np.empty(swpressure.size)
swtemp_filt = np.empty(swtemp.size)
dst_filt = np.empty(dst.size)

freq_ranged = 5/365
freq_rangey = 75/365

print(1/dom_freqs[0]/365)
print(1/dom_freqs[12])
print(1/dom_freqs[8])

newarray = [swavgB_filt, swdensity_filt, swvelocity_filt, swpressure_filt, swtemp_filt, dst_filt]
# use ifft to filter out dominant frequencies found above
for i in range(6):
    x = interpolate_nan(arrays[i])
    N = x.size
    amps = fft(x)
    freqs = fftfreq(N, 1/24)

    mask1 = (np.abs(freqs) < dom_freqs[0]+freq_rangey) & (np.abs(freqs) > dom_freqs[0]-freq_rangey) & (freqs != 0)
    mask2 = (np.abs(freqs) < dom_freqs[12]+freq_ranged) & (np.abs(freqs) > dom_freqs[12]-freq_ranged) & (freqs != 0)  
    mask3 = (np.abs(freqs) < dom_freqs[8]+freq_ranged) & (np.abs(freqs) > dom_freqs[8]-freq_ranged) & (freqs != 0) 

    amps_filt = amps.copy()
    amps_filt[mask1]=0
    amps_filt[mask2]=0
    amps_filt[mask3]=0
    newarray[i] = ifft(amps_filt)

swavgB_filt = newarray[0]
swdensity_filt = newarray[1]
swvelocity_filt = newarray[2]
swpressure_filt = newarray[3]
swtemp_filt = newarray[4]
dst_filt = newarray[5]

# %% [markdown]
# Plots the new data with the dominant frequencies filtered out from the ifft band pass filter.
#

# %%
# correlates data to labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB,swavgB_filt),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity,swvelocity_filt),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure,swpressure_filt),
        'Plasma Temperature|Temperature (K)': (time,swtemp,swtemp_filt),
        'Ion Number Density|Density (per cc)': (time,swdensity,swdensity_filt),
        'DST Index|DST (nT)': (time,dst,dst_filt)}
fig, axes = plt.subplots(2,3, figsize = (15,8))
# for loop to add data to each plot
for ax, (label, (x, y,filt)) in zip(axes.flat, data.items()):
    #add data to the plot
    ax.plot(x,y, label = 'Original Data', color = '#3477eb',alpha = 0.75)
    ax.plot(x, filt, label = 'Filter Data', color = '#bd1caa', alpha = 0.75)
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel(r'Date $(year)$')
    ax.set_ylabel(ytext)
    ax.legend()
fig.tight_layout()

# %% [markdown]
# The solar wind speed appears to be the most impacted by the dominant frequencies removed as it shows the change from the original to the filtered data
#

# %% [markdown]
# ## Data Analysis
# *takes a while to load

# %% [markdown]
# ### Identifying DST Events

# %%
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import math as math

# %%
#every hour
dst_events = np.zeros(len(dst))
for i in range(len(dst)):
    if(dst[i] < -70):
        dst_events[i] = True

# %%
#Here we will create an array to hold our true
dst_binary0 = np.zeros(math.ceil(len(time) / 120))
window_size0 = timedelta(days=5)
start, stop = time[0], time[-1]
idx = 0
while start + window_size0 < stop:
    end = start + window_size0
    locations = (time >= start) & (time < end)
    subset = dst[locations]
    if(np.min(subset) < -75):
        dst_binary0[idx] = True
    start += window_size0
    idx += 1

# %% [markdown]
# ### Identifying Events in the Filtered Solar Wind

# %%
swEvent = np.empty(swavgB_filt.size)

for i in range(swavgB_filt.size):
    c = 0
    if swavgB_filt[i] > 15:
        c = c+1
    if swdensity_filt[i] > 15:
        c = c+1
    if swpressure_filt[i] > 10:
        c = c+1
    if swtemp_filt[i] > 7.5*10**5:
        c = c+1
    if swvelocity_filt[i] > 550:
        c = c+1
    if c >=3:
        swEvent[i] = True
    else:
        swEvent[i] = False


# %%
#here we will perform our binary event analysis on the Solar Wind data starting with the same cutoffs as chosen above
#this time however, we will us a while loop and datetime objects to retrieve 3 day time intervals instead of
#checking each data point individually
#Here we will create an array to hold our true
sw_binary0 = np.zeros(math.ceil(len(time) / (120)))
window_size0 = timedelta(days=5)
start, stop = time[0], time[-1]
idx = 0
count = 0
while start + window_size0 < stop:
    end = start + window_size0
    locations = (time >= start) & (time < end)
    subset = dst[locations]
    c = 0
    if np.max(swavgB_filt[locations]) > 15:
        c = c+1
    if np.max(swdensity_filt[locations]) > 15:
        c = c+1
    if np.max(swpressure_filt[locations]) > 10:
        c = c+1
    if np.max(swtemp_filt[locations]) > 5*10**5:
        c = c+1
    if np.max(swvelocity_filt[locations]) > 550:
        c = c+1
    if c >=3:
        sw_binary0[idx] = True
    else:
        sw_binary0[idx] = False
    start += window_size0
    idx += 1


# %% [markdown]
# ### Re-code binary list calculations within a function for repeatability

# %%
#Here we define a function which will record the true/false values for our DST data
def calc_dst_binary(window_size, cutoff):
    '''Creates a true or false array for whether or 
    not a dst event exists in each time interval
    window is the window size in days
    cutoff is the dst index below which we count an event, in units of nano-Teslas

    Returns: binary event T/F array for dst data
    '''
    dst_binary = np.zeros(math.ceil(len(time) / (window_size * 24)))
    window = timedelta(days=window_size)
    start, stop = time[0], time[-1]
    idx = 0
    while start + window < stop:
        end = start + window
        locations = (time >= start) & (time < end)
        subset = dst[locations]
        if(np.min(subset) < cutoff):
            dst_binary[idx] = True
        start += window
        idx += 1
    return dst_binary


# %%
#Here we define a function which will record the true/false values for our solar wind data
def calc_sw_binary(window_size, cutoffs):
    '''Creates a true or false array for whether or 
    not a sw event exists in each time interval
    window is the window size in days
    cutoffs is an array that gives the cutoffs for each variable in the following order:
    [swavgB_filt, swdensity_filt, swpressure_filt, swtemp_filt, swvelocity_filt]

    Returns: binary event T/F array for sw data
    '''
    sw_binary = np.zeros(math.ceil(len(time) / (window_size * 24)))
    window = timedelta(days=window_size)
    start, stop = time[0], time[-1]
    idx = 0
    count = 0
    while start + window < stop:
        end = start + window
        locations = (time >= start) & (time < end)
        subset = dst[locations]
        c = 0
        if np.max(swavgB_filt[locations]) > cutoffs[0]:
            c = c+1
        if np.max(swdensity_filt[locations]) > cutoffs[1]:
            c = c+1
        if np.max(swpressure_filt[locations]) > cutoffs[2]:
            c = c+1
        if np.max(swtemp_filt[locations]) > cutoffs[3]:
            c = c+1
        if np.max(swvelocity_filt[locations]) > cutoffs[4]:
            c = c+1
        if c >=3:
            sw_binary[idx] = True
        else:
            sw_binary[idx] = False
        start += window
        idx += 1
    return sw_binary


# %%
#calculate the T/F array for events in the dst data
window_size = 5 #days
cutoff = -70 #nT
dst_binary = calc_dst_binary(window_size, cutoff)

# %%
#calculate the T/F array for events in the sw data
cutoffs = [15, 15, 10, 7.5*10**5, 550]
sw_binary = calc_sw_binary(window_size, cutoffs)

# %%
#create figure and suplots to plot results
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2, figsize=(10, 8))
#dst events by hour
ax1.plot(time, dst_events, color = '#bd1caa')
#dst events 5 day window
ax2.plot(dst_binary, color = '#bd1caa')
#solar wind events by hour
ax3.plot(time, swEvent, color = '#bd1caa')
#solar wind events 5 day window
ax4.plot(sw_binary, color = '#bd1caa')
# adds proper titles and labels
ax1.set_title(f'Events Identified from Hourly DST Data\nTotal Events Identified: {int(sum(dst_events))}')   
ax1.set_xlabel(r'Date $(year)$')
ax1.set_ylabel('1 = Solar Event and 0 = No Event')

ax2.set_title(f'Events Identified from DST Data, 5 day window\nTotal Events Identified: {int(sum(dst_binary))}')   
ax2.set_xlabel('5 day window')
ax2.set_ylabel('1 = Solar Event and 0 = No Event')

ax3.set_title(f'Events Identified from Hourly Solar Wind Data\nTotal Events Identified: {int(sum(swEvent))}')   
ax3.set_xlabel(r'Date $(year)$')
ax3.set_ylabel('1 = Solar Event and 0 = No Event')

ax4.set_title(f'Events Identified from Solar Wind Data, 5 day window\nTotal Events Identified: {int(sum(sw_binary))}')   
ax4.set_xlabel('5 day window')
ax4.set_ylabel('1 = Solar Event and 0 = No Event')
print(f'Total event idendified from solar wind data: {int(np.sum(swEvent))}')

fig.tight_layout()

# %% [markdown]
# ## here is binary event anaylsis on two lists 

# %%
## here is binary event anaylsis on two lists that we can edit later --- just wanted to have something before wed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def binary_event_analysis(list1, list2):
    """
    Analyzes two binary event lists.
    
    Parameters:
    - list1, list2: Lists of binary values of equal length.
    
    This function computes:
      - A contingency table for the two lists.
      - The phi coefficient (correlation).
      - The odds ratio.
      - The hit rate, false alarm rate, proportion correct, and false alarm ratio.
      - The Heidke Skill Score (HSS) for forecast skill.
      - Chi-square test for independence.
    """
    # Inputs are of equal length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Construct the contingency table elements
    # a: True Positives, b: False Alarms, c: Misses, d: True Negatives.
    a = sum(1 for i, j in zip(list1, list2) if i == 1 and j == 1)
    b = sum(1 for i, j in zip(list1, list2) if i == 1 and j == 0)
    c = sum(1 for i, j in zip(list1, list2) if i == 0 and j == 1)
    d = sum(1 for i, j in zip(list1, list2) if i == 0 and j == 0)
    
    contingency_table = np.array([[a, b],
                                  [c, d]])
    
    # Total number of observations
    N = a + b + c + d
    # Print contingency table
    print("Contingency Table:")
    print("                list2=1   list2=0")
    print(f"list1=1        {a:<9} {b}")
    print(f"list1=0        {c:<9} {d}\n")
    
    # Compute phi coefficient
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = numerator / denominator if denominator != 0 else np.nan
    print(f"Phi coefficient (correlation): {phi:.4f}")
    
    # Calculate Odds Ratio
    if b * c == 0:
        odds_ratio = np.inf if a * d > 0 else np.nan
        print("Odds Ratio: Division by zero occurred (one of b or c is 0); odds ratio set to infinity if numerator > 0.")
    else:
        odds_ratio = (a * d) / (b * c)
        print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Calculate additional performance metrics
    # Hit Rate: Proportion of actual positive events (list2) that were correctly forecast
    hit_rate = a / (a + c) if (a + c) != 0 else np.nan
    # False Alarm Rate: Proportion of actual negative events (list2) that were falsely forecast as positive.
    false_alarm_rate = b / (b + d) if (b + d) != 0 else np.nan
    # Proportion Correct: Overall accuracy
    proportion_correct = (a + d) / N if N != 0 else np.nan
    # False Alarm Ratio: Proportion of forecasted positives that were false alarms
    false_alarm_ratio = b / (a + b) if (a + b) != 0 else np.nan

    Precision = (a) / (a + b) if (a) != 0 else np.nan
    Recall = (a) / (a + c) if (a) != 0 else np.non
    
    print(f"\nHit Rate (True Positive Rate): {hit_rate:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    print(f"Proportion Correct (Overall Accuracy): {proportion_correct:.4f}")
    print(f"False Alarm Ratio: {false_alarm_ratio:.4f}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")

    # Calculate Heidke Skill Score (HSS)
    # Expected accuracy by chance
    Pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (N * N) if N != 0 else np.nan
    # HSS: (observed accuracy - expected accuracy) / (1 - expected accuracy)
    HSS = (proportion_correct - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan
    print(f"Heidke Skill Score: {HSS:.4f}")
    

    plt.figure(figsize=(6, 4))
    
    plt.imshow(contingency_table, cmap="RdPu", vmin=0, vmax=200)

    #adding a colorbar to display the mapping of colors to numerical values.
    plt.colorbar()

    plt.title('Map of Confusion Matrix')
    plt.xlabel('Observed Values (DST Index)')
    plt.ylabel('Forecasted Values (Solar Wind)')

    plt.xticks(ticks=[0, 1], labels=['1', '0'])
    plt.yticks(ticks=[0, 1], labels=['1', '0'])

    #get the numeric values
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            plt.text(j, i, str(contingency_table[i, j]),
                     ha="center", va="center", color="black")

    plt.show()

    # Return all results as a dictionary
    return {
        'contingency_table': contingency_table,
        'phi_coefficient': phi,
        'odds_ratio': odds_ratio,
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'proportion_correct': proportion_correct,
        'false_alarm_ratio': false_alarm_ratio,
        'heidke_skill_score': HSS,
        'Precision': Precision, 
        'Recall': Recall
    }


# %%
#try out the binary analysis function, natalie please feel free to fix this part up and do it properly later!
binary_event_analysis(dst_binary, sw_binary)
print('\n yay!  we have some stats! *high five*')

# %%
import math
from datetime import timedelta
from scipy.stats import qmc


window_day = timedelta(days=1) #start with 1 day window
start, stop = time[0], time[-1] #start and end of data set
n_windows_1d = math.ceil((stop - start) / window_day) #how many one day windows fit in the period
W = np.zeros((n_windows_1d, 6)) #empty array to hold all the vals

for i in range(n_windows_1d):  #loop over each day-index from 0 to the number of windows
    t0 = start + i * window_day
    t1 = t0 + window_day #get the start and stop of each time slice
    mask = (time >= t0) & (time < t1) #find the points within this slice
    
    #compute the summary numbers for that day
    #use .real because of a warning error about discarding imaginary parts
    W[i, 0] = np.min(dst[mask]) #picks the dst vals that fall in that day, and then take the smallest
    W[i, 1] = np.max( swavgB_filt[mask].real )
    W[i, 2] = np.max( swdensity_filt[mask].real ) #grab the vals, but the largest
    W[i, 3] = np.max( swpressure_filt[mask].real )
    W[i, 4] = np.max( swtemp_filt[mask].real )
    W[i, 5] = np.max( swvelocity_filt[mask].real )


#helper function to take window size, dst threshold, and array of sw vals and return a boolian to
#say if the event occured or not
def compute_events(window_days, dst_cut, sw_cuts, sw_thresh=3):
    step = window_days
    n_int = math.ceil(n_windows_1d / step) #find how many sections of the windowed days fit in the whole space
    dst_ev = np.zeros(n_int, dtype=bool) #empty list to hold vals (bool)
    sw_ev = np.zeros(n_int, dtype=bool) #empty list to hold vals (bool)
    for j in range(n_int): #loop through num of sections
        seg = W[j*step:(j+1)*step] #grab a chunck of the rows
        dst_ev[j] = np.min(seg[:, 0]) < dst_cut #use the daily min dst col to find if the val is below min
        agg_max = np.nanmax(seg[:, 1:], axis=0) #get columns 1-5, ignore n/as and get the cols max
        sw_ev[j] = np.sum(agg_max > sw_cuts) >= sw_thresh #find the peaks that exceed the cutoffs
    return dst_ev, sw_ev

#convert the booleans into hit rate/false alarm rate
def get_rates(dst_ev, sw_ev):
    a = np.sum(dst_ev & sw_ev)
    b = np.sum(dst_ev & ~sw_ev)
    c = np.sum(~dst_ev & sw_ev)
    d = np.sum(~dst_ev & ~sw_ev)
    hit_rate = a / (a + c) if (a + c) > 0 else np.nan
    false_alarm_rate = b / (b + d) if (b + d) > 0 else np.nan
    return hit_rate, false_alarm_rate

#set the parameter ranges that we are going to check
param_lists = {
    'window':   [1, 3, 5, 7, 10],
    'dst':      [-70, -150, -330],
    'B':        [5, 15, 30],
    'density':  [5, 15, 25],
    'pressure': [5, 10, 20],
    'temp':     [3e5, 7.5e5, 1e6],
    'velocity': [300, 550, 800]
}

#latin hypercube sampling -- parameter range is sampled evenly, but only 200 samples are taken
keys = list(param_lists.keys()) #hold parameters
sampler = qmc.LatinHypercube(d=len(keys)) #do the latinhypercube
samples = sampler.random(n=200) #hold samples

hrs, fars, combos = [], [], []
#convert the noramlized sample into the closet index in the parameter list
#build 5-var cuttoff array
#make boolean series
#get hit rate/false alarm rate
#store
for row in samples: #loop through latin hypercube sampling output
    combo = {}
    for idx, key in enumerate(keys): #for each key look up the possible values
        choices = param_lists[key]
        sel = min(int(row[idx] * len(choices)), len(choices)-1) #scale the normalized sample
        combo[key] = choices[sel]
    sw_cuts = np.array([combo['B'], combo['density'],
                        combo['pressure'], combo['temp'],
                        combo['velocity']]) #get the combo
    dst_ev, sw_ev = compute_events(combo['window'], combo['dst'], sw_cuts) #count the events
    hr, far = get_rates(dst_ev, sw_ev) 
    hrs.append(hr)
    fars.append(far)
    combos.append(combo)

#find the top three combos based on lowest false alarm rate and highest hit rate
ranked = sorted(
    zip(hrs, fars, combos),
    key=lambda x: (-x[0], x[1])
)
top3 = ranked[:3]

##make a single roc inspired plot that has the top 3 labeled
plt.figure(figsize=(7,7))
plt.scatter(fars, hrs, alpha=0.6, color = 'hotpink')
plt.plot([0, 1], [0, 1], 'k--', lw=0.8)

#annotate top 3 on graph
for rank, (hr, far, combo) in enumerate(top3, start=1):
    plt.annotate(str(rank), (far, hr),
                 textcoords="offset points", xytext=(5,-5))

plt.xlabel('False Alarm Rate')
plt.ylabel('Hit Rate')
plt.title('ROC: LHS Samples (Top‑3 Highlighted)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 8)print the top 3 combos
print("Top 3 parameter combos (high HR, low FAR):")
for rank, (hr, far, combo) in enumerate(top3, start=1):
    print(f"\nRank {rank}:")
    print(f" Window size: {combo['window']} days")
    print(f" DST cutoff: {combo['dst']} nT")
    print(f" B cutoff: {combo['B']}")
    print(f" Density cutoff: {combo['density']}")
    print(f" Pressure cutoff: {combo['pressure']}")
    print(f" Temp cutoff: {combo['temp']}")
    print(f" Velocity cutoff: {combo['velocity']}")
    print(f" Hit Rate: {hr:.3f}")
    print(f" False Alarm Rate: {far:.3f}")

# %% [markdown]
# ## Conclusions
# Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# ## References
#

# %%
#Here is a ROC curve along with some analysis: 
from sklearn.metrics import roc_curve, auc
from datetime import timedelta

#ROC Curve Function 
def plot_roc_from_scores(true_labels, condition_scores):
    """
    Generates and plots an ROC curve from binary true labels and discrete prediction scores.
    
    Parameters:
    - true_labels: Array-like of 0s and 1s representing actual event occurrences (e.g., dst_binary)
    - condition_scores: Array-like of integers (0–5) representing the number of solar wind thresholds met

    Output:
    - Displays ROC curve and prints AUC.
    """
    y_true = np.array(true_labels)
    y_scores = np.array(condition_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='pink', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Solar Wind Event Forecasting')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"AUC (Area Under Curve): {roc_auc:.4f}")
    return 


sw_score = np.zeros(math.ceil(len(time) / 120))
window_size = timedelta(days=5)
start, stop = time[0], time[-1]
idx = 0

while start + window_size < stop:
    end = start + window_size
    locations = (time >= start) & (time < end)
    
    c = 0
    if np.max(swavgB_filt[locations]) > 15:
        c += 1
    if np.max(swdensity_filt[locations]) > 15:
        c += 1
    if np.max(swpressure_filt[locations]) > 10:
        c += 1
    if np.max(swtemp_filt[locations]) > 7.5 * 10**5:
        c += 1
    if np.max(swvelocity_filt[locations]) > 550:
        c += 1

    sw_score[idx] = c  # Store the score (0–5)
    start += window_size
    idx += 1

plot_roc_from_scores(dst_binary, sw_score)

# %% [markdown]
#

# %% [markdown]
#
