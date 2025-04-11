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
data = np.genfromtxt('OMNI2_20002024.csv', names=True, delimiter=',', skip_header=97, encoding='utf-8',converters={0:tconvert}, dtype=None)

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
fig.suptitle('Power Spectrums')

dom_amps = []
dom_freqs = []

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
    for i in range(10):
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


fig.tight_layout()

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

newarray = [swavgB_filt, swdensity_filt, swvelocity_filt, swpressure_filt, swtemp_filt, dst_filt]
# use ifft to filter out dominant frequencies found above
for i in range(6):
    x = interpolate_nan(arrays[i])
    N = x.size
    amps = fft(x)
    freqs = fftfreq(N, 1/24)

    mask1 = (freqs == dom_freqs[0])  
    mask2 = (freqs == dom_freqs[3])  
    mask3 = (freqs == dom_freqs[3])  
    mask4 = (freqs == dom_freqs[8]) 
    mask5 = (freqs == dom_freqs[25])

    amps_filt = amps.copy()
    amps_filt[mask1]=0
    amps_filt[mask2]=0
    amps_filt[mask3]=0
    amps_filt[mask4]=0
    amps_filt[mask5]=0
    newarray[i] = ifft(amps_filt)

swavgB_filt = newarray[0]
swdensity_filt = newarray[1]
swvelocity_filt = newarray[2]
swpressure_filt = newarray[3]
swtemp_filt = newarray[4]
dst_filt = newarray[5]

# %%
# correlates data to labels
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB,swavgB_filt),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity,swvelocity_filt),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure,swpressure_filt),
        'Plasma Temperature|Temperature (K)': (time,swtemp,swtemp_filt),
        'Ion Number Density|Density (per cc)': (time,swdensity,swdensity_filt),
        'DST Index|DST (nT)': (time,dst,dst_filt)}
fig, axes = plt.subplots(6,1, figsize = (12,20))
# for loop to add data to each plot
for ax, (label, (x, y,filt)) in zip(axes.flat, data.items()):
    #add data to the plot
    ax.plot(x,y, label = 'Original Data')
    ax.plot(x, filt, label = 'Filter Data')
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel(r'Date $(year)$')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %%
from scipy.stats import chi2_contingency



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
    
    print(f"\nHit Rate (True Positive Rate): {hit_rate:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    print(f"Proportion Correct (Overall Accuracy): {proportion_correct:.4f}")
    print(f"False Alarm Ratio: {false_alarm_ratio:.4f}")
    
    # Calculate Heidke Skill Score (HSS)
    # Expected accuracy by chance
    Pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (N * N) if N != 0 else np.nan
    # HSS: (observed accuracy - expected accuracy) / (1 - expected accuracy)
    HSS = (proportion_correct - Pe) / (1 - Pe) if (1 - Pe) != 0 else np.nan
    print(f"Heidke Skill Score: {HSS:.4f}")
    
    # Chi-Square test for independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-Square Test for Independence:")
    print(f"Chi2 statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4f}")
    print("Expected frequencies:")
    print(expected)
    
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
        'chi2_statistic': chi2,
        'chi2_dof': dof,
        'chi2_p_value': p,
        'expected_frequencies': expected
    }
