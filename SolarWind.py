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
# # **Solar WIN(d)**
#
# ### Elise Segal, Daniel Heilmen, Natalie Giovi, Percy Slattery
#
# ## Introduction & Approach
# Add a brief description of the goal and background knowledge for the lab. This can be drawn from the lab description, but should be in your own words.
#
# ## Data
# List the datasets used, what they describe and any quality/pre-processing before analysis.
#
# ----
#

# %% [markdown]
# # Import Relevant Libraries  

# %%
# imports the libraries needed for the project
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
import math
from scipy.stats import qmc

# %% [markdown]
# # Data Collection    
#
# ## Selection   
# Start by importing the data from OMNI. We used a csv file and data from 2000 to 2024 to show 2 solar cycles. From OMNI we chose hourly data because trying to get minute data from OMNI takes too long to upload into a notebook. From OMNI, we retrieved time of the measurement, the average magnetic field at 1 AU, the solar wind speed at 1 AU, the solar wind pressure at 1 AU, the solar wind temperature at 1 AU, the solar wind density at 1 AU and the DST Index from stations around the equator on the surface of the Earth.

# %%
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
# ## Cleaning Real World Data   
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
# ## Organizing and Understanding our Data
# First step is to visualize the data to ensure some periodic behavior exists for the fft. This is done by creating a dictionary to correlate all of the data with proper labeling for the graphs and making all the plots through a for loop.

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
# While the most notable solar events from the plots during solar maximums, it is not the best example to show how solar wind parameters behave during a CME. This is because the they are major storms that could have been the result of multiple CMEs back to back (cannibal CME). Therefore we decided to go with the April 23rd, 2023 storm which was the result on a single CME to showcase how solar wind changes during a solar event. This was plotted using the same technique as for plotting the entire dataset but then added a x axis limit to focus on the dates around the 4/23/23 storm.

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
# # Removing Solar Wind Periodicities
#
# ## FFT Analysis   
# Next step is to do the fft however the fft cannot have any NaN values. To start we created an interpolate function that replaces any NaN values with values similar that are around it allowing us to proceed with the fft and ifft later. Then we created a function that takes the fft and returns arrays with the frequencies and power from the fourier transform to plot the power spectrums. This was done with the scipy fftpack learned in class.

# %%
# Helper function: interpolate missing (NaN) values in the time series using linear interpolation
def interpolate_nan(y):
    '''
    This function interpolate missing (NaN) values in the time series using linear interpolation so the fft and ifft can be computed.
    
    Parameters
    ----------
    y:array
        The array that needs to be interpolated
        
    Returns
    -------
    y: array
        The final array with interpolated values and no NaN values.
 
    '''
    nans = np.isnan(y)                          # Identify NaN positions
    not_nans = ~nans                            # Identify valid (non-NaN) positions
    x = np.arange(len(y))                       # Create an index array
    y[nans] = np.interp(x[nans], x[not_nans], y[not_nans])  # Interpolate NaNs
    return y

def analyze_fft(data, dt_hours=1, top_k=6):
    '''
    This functions calculates the fft for the data according to the given time period and then returns all the need information from the fft to later make a power spectrum plot.
    
    Parameters
    ----------
    data:array
        The array of data to calculate the fft from 
    dt_hours:int
        The time step of the data to get the right units for the frequencies - set at 1 for 1 day in 24 to get cycles per day
    top_k:int
        The number of top harmonics to be recorded. 
        
    Returns
    -------
    freq: array
        An array with all the frequencies retrieved from the fourier transform.
    power:array
        The powers corresponding to all the frequencies from the fourier transform 
    n:int
        The length of the data array
    harmonics:array
        Array of dictionaries with the top top_k frequencies and the corresponding amplitude, period in days and the harmonic index.  
 
    '''
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
    '''
    Sorts the harmonic array generated from the analyze_fft to be in order of highest amplitude
    
    Parameters
    ----------
    harmonics:array
        The array of harmonics to be sorted
        
    Returns
    -------
    sorted_h: array
        The array of sorted harmonics
 
    '''
    sorted_h = sorted(harmonics, key=lambda h: h['amplitude'], reverse=True)
    return sorted_h


# %% [markdown]
# ## Finding the Dominant Frequencies
# Plots the power spectrums for each parameter and finds the top 5 dominant frequencies in each parameter. The dominant frequencies are found by finding the max power and then the corresponding frequency and adding it to an array then deleting that value and finding the new max. This is done using the same method of plotting with a dictionary correlating labels with the data and a for loop. The fft is preformed in the for loop that is then plotted for the power spectrums.
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

# %% [markdown]
# From the power spectrums several dominant frequencies are observed. Specifically, there are several peaks showing high power frequencies for the solar wind velocity, temperature and ion number density. The remain power spectrums do not showcase as prevelant dominant frequencies in their power spectrums. Of the ones clearly showing several dominant frequencies, there are around 5 well define peaks for dominant frequencies that appear to be in the same place as the other parameters.

# %% [markdown]
# To get the specific dominant frequencies, the sorted_amps function gets the the top 6 the output isn't formatted in the best way for readablility. So still using a for loop to go through the dictionary, the top five dominant frequencies are found by finding the max power and corresponding frequency and then removing it to find the next max for dominant frequencies. The dominant frequencies is then printed in cycles per day and then the corresponding period in days and years for better understanding of the dominant cycles shown in the solar wind. *Note: Does print a lot of lines so it might be truncated, switch to scrollable element to see full output of the print statements.*

# %%
#arrays for storing the dominant frequencies
dom_amps = []
dom_freqs = []

# for loop to loop through all the parameters in the previous defined dictionary of data
for (label, (x, i)) in data.items():
    freq, power, n, harmonics = analyze_fft(i)
    sorted_amps = sort_harmonics_amp(harmonics)
    freq = abs(freq)
    # prints values from sorted amps
    print(sorted_amps)
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

    # Print the results in easy read format better than sorted_amps
    print(f'\n{title}:')
    for rank, freq in enumerate(dominant_frequencies, start=1):
        days = 1 / freq
        years = days / 365
        print(f'\t {rank} Dominant frequency: {freq:.4f} cycles per day (~{days:.2f} days or ~{years:.4f} years)')
    print('')

# %% [markdown]
# Dominant frequencies that are repeated through multiple of the parameters are 12 years, 27 days and 9 days. The 12 years correlates to the solar cycle and the 27 days correlates to the solar rotation cycle so these as dominant frequencies is expected.
#

# %% [markdown]
# ## Removing the Dominant Frequencies
#
# Removes the dominant frequencies using a band pass filter and ifft. Removes all frequencies around 5 days of the 2 dominant frequencies in days and 3 months for the dominant frequency in years. First, new arrays are created to put in the new filtered data from the arrays. Then a for loop is used to remove the 3 frequencies and the ranges around them using masks after interpolating the data to ensure no NaN values. During the fft, the data is not normalized so the ifft can then be taken directly and the new filtered data is added to the empty arrays created in the beginning.
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
# Plots the new data with the dominant frequencies filtered out from the ifft band pass filter. This is done using an updated dictionary with the new filtered values and a for loop to add the data labels to all the plots. The new filtered data is overlayed with the original data from OMNI to show the difference after removing the dominant frequencies.
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
# The solar wind speed appears to be the most impacted by the dominant frequencies removed as it shows the change from the original to the filtered data. Overall, the filtered data condenses around the mean more.
#

# %% [markdown]
# ## Data Interpretation: Identifying Solar Events
#
# We will be using binary event analysis in order to measure the effectiveness of our model in identifying events in the solar wind.  However after filtering out our dominant frequencies, we are only left with a filtered spectra, and do not yet have a clear notion of when this spectra indicates that an event is occurring.  In order to perform binary event analysis on our data, we need a method for determining what constitutes an event detection in our spectra.  We can then apply this method to identify when our spectra predicts that one occurs, and apply this method to record, for a list of time intervals, whether there was an event during each time window.  In this way we reduce a complex set of filtered data into a simple binary “true” or “false”, hence the name “binary event analysis”.  
#
# We can compare our list identified with filtered solar wind data to a similar binary list for identical time intervals from the DST index data in order to assess how well our method was able to identify real events.  For the purposes of this analysis, we treat the DST index data as a reliable indicator of when solar events occurred, and the binary list of events identified from the DST data serves as our list of “observed” events in our binary analysis, against which we compare our “predicted” events identified from our solar wind data.  
#
# In order to determine whether or not an event occurred, we will use a set of cutoffs, with one for each of our six parameters.  When our data spikes above its respective threshold, or in the case of the DST data, drops below, we flag that as an event.  For our five solar wind parameters, we will consider there to have been a detection if three or more of the parameters are in agreement.  The challenge then is to select appropriate cutoffs for each of our parameters.  We will select a few values as a starting point for our analysis and, as explained later, move on to explore the parameter space a bit further and investigate how changes to the cutoffs and other parameters influence the performance of our event identification method.  
#
# For the DST data, we turn to the literature for a suggested cutoff below which there is a storm indicative of a solar event.  Palacios et. al. 2018 lists values of -75 nT, -150 nT, and -330 nT as common values of thresholds for moderate, intense, and extreme geomagnetic storms respectively.  We will start by assessing our results using the -75 nT threshold for the DST index.  
#
# For our solar wind parameters, our initial intuition was to turn to the literature as well; however, since we subtracted out our dominant frequencies, our filtered data is no longer representative of the physical values.  Thus, we instead examined the April, 2023 CME from the "Organizing and Understanding our Data" section, and selected starting values for our solar wind parameter cutoffs based on this event.  The initial cutoffs we used were the following: 15 nT for the average magnetic field, 15 ions per cc for the ion number density, 550 km/s for the solar wind velocity, 10 nPa for the solar wind pressure, and 750,000 K for the solar wind temperature.

# %% [markdown]
# ### Identifying Events Hourly
#
# Determine whether or not an event is occurring for each index one by one.
#
# Initially, we used a window size of one hour.  In other words, we simply check each index and compile a list for each index of whether or not there is an event.  However, there are a couple of issues with this.  Firstly, it would be erroneous to treat events which last for more than one hour as a number of distinct, separate ones.  Secondly and most importantly, not all of the parameters are affected equally by an event.  Namely, the average magnetic field, solar wind velocity, and DST index are affected over a longer period of time than plasma flow pressure, plasma temperature, and ion number density which are affected over a shorter period.  Thus, there are indices for which the DST index is still affected, only two out of five of the solar wind parameters are still impacted by the event, and our comparison will suggest that our method failed to identify a solar event, as indicated by the DST index.

# %%
#every hour
dst_events = np.zeros(len(dst))
for i in range(len(dst)):
    if(dst[i] < -75):
        dst_events[i] = True

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


# %% [markdown]
# ### Windowed Approach
# To solve this issue, we advanced to a longer window length, starting here with a 5 day window.  In the windowed approach, we break the time period up into equal length windows (in this case 5 days each), and step through each window, starting with the beginning of our time period.  If the parameter crossed the cutoff at some point during the window, we consider that an indication of an event.  In the case of the solar wind data, we require at least three or more of the parameters to have crossed the cutoff in order for an event to be detected.  For each window we record whether there was (True) or wasn't (False) an event, for both the dst and solar wind values.

# %%
def calc_dst_binary(window_size, cutoff):
    '''Creates a true or false array for whether or 
    not a dst event exists in each time interval

    Parameters:
        window_size is the window size in days
        cutoff is the dst index below which we count an event, in units of nano-Teslas

    Returns: binary event T/F array for dst data
    '''
    #create an empty array to hold our t/f values
    dst_binary = np.zeros(math.ceil(len(time) / (window_size * 24)))
    #set our window variable equal to a datetime range of 5 days
    window = timedelta(days=window_size)
    start, stop = time[0], time[-1]
    idx = 0
    #use while loop to step through time period and find values for all intervals
    while start + window < stop:
        #define the end of the current interval
        end = start + window
        #filter the dst data to grab only the indices within our current window
        locations = (time >= start) & (time < end)
        subset = dst[locations]
        #determine if there is a storm or not based on the minimum value and cutoff
        if(np.min(subset) < cutoff):
            dst_binary[idx] = True
        #update the window start time and array index
        start += window
        idx += 1
    return dst_binary


# %%
def calc_sw_binary(window_size, cutoffs):
    '''Creates a true or false array for whether or 
    not a sw event exists in each time interval

    Parameters:
        window_size is the window size in days
        cutoffs is an array that gives the cutoffs for each variable in the following order:
        [swavgB_filt (nT), swdensity_filt (km/s), swpressure_filt (nPa), swtemp_filt (K), swvelocity_filt (n per cc)]

    Returns: binary event T/F array for sw data
    '''
    #create an empty array to hold our t/f values
    sw_binary = np.zeros(math.ceil(len(time) / (window_size * 24)))
    #create window variable and set start and stop as the beginning and end of the time period
    window = timedelta(days=window_size)
    start, stop = time[0], time[-1]
    idx = 0
    #use while loop to step through time period and find values for all intervals
    while start + window < stop:
        #define the end of the current interval
        end = start + window
        locations = (time >= start) & (time < end)
        #filter the data to grab only the indices within our current window and
        #determine whether there is an event within the window for each parameter
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
        #if 3 or more of the 5 paramters agree with one another, flag as an event
        if c >=3:
            sw_binary[idx] = True
        else:
            sw_binary[idx] = False
        #update the window start time and array index
        start += window
        idx += 1
    return sw_binary


# %% [markdown]
# #### Identify and Plot Events:

# %%
#calculate the T/F array for events in the dst data
window_size = 5 #days
cutoff = -75 #nT
dst_binary = calc_dst_binary(window_size, cutoff)

# %%
#calculate the T/F array for events in the sw data
cutoffs = [15, 20, 10, 7.5*10**5, 550]
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
# In total, we have identified 259 events in the solar wind data.  This number is much more reasonable than the number identified when we treated each indice past the cutoff as a separate event.  It is important to note a flaw which still remains in this method of dividing up the time period.  It is still possible, for whatever window size is chosen, for multiple events to fall into one window and be counted as one, for one event to be split between two windows and be counted separately, or for an event to occur between windows such that the solar wind data identifies the event as having occurred in a different window than the dst data, such that it is counted as a false alarm and a miss instead of a hit.  While a few occurances like these may occur, we will consider this to be a minor issue to have a negligible effect.  In addition, we will experiment with a range of window sizes to vary our results.

# %% [markdown]
# # Data Analysis: Binary Event Analysis
#
# ## What is Binary Event Analysis and Why is it Applicable?
#
# Binary Event Analysis takes two arrays that are both in binary form and compares them, creating a matrix of values organized into sections of True positives, False positives, True negatives, and False negatives. By creating this matrix, the user is able to compare the two lists and determine correlation. 
#
# This type of analysis great for simplifying the data and classifying the prediction of extreme events. This was very applicable to our analysis because we wished to see just how good our data was at predicting events once the periodicities were removed. Binary event analysis also allows for the use of Receiver Operating Characteristic style analysis, which compares True positives to False positives. This topic will be discussed in detail later in the document.
#
# ## The Function and its Functionality
#
# ADD ANALYSIS HERE

# %%
def binary_event_analysis(list1, list2, printit = True):
    """
    Analyzes two binary event lists.
    
    Parameters:
    - list1, list2: Lists of binary values of equal length.
    - if all calculation and tables should automatically be printed
    
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
    if printit:
        print("Contingency Table:")
        print("                list2=1   list2=0")
        print(f"list1=1        {a:<9} {b}")
        print(f"list1=0        {c:<9} {d}\n")
    
    # Compute phi coefficient
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = numerator / denominator if denominator != 0 else np.nan
    if printit:
        print(f"Phi coefficient (correlation): {phi:.4f}")
    
    # Calculate Odds Ratio
    if b * c == 0:
        odds_ratio = np.inf if a * d > 0 else np.nan
        if printit:
            print("Odds Ratio: Division by zero occurred (one of b or c is 0); odds ratio set to infinity if numerator > 0.")
    else:
        odds_ratio = (a * d) / (b * c)
        if printit:
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
    
    if printit:
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
    if printit:
        print(f"Heidke Skill Score: {HSS:.4f}")
    
    if printit:
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


# %% [markdown]
# ## Binary Event Analysis: Our Data's Original Output
#
# Once the values of both arrays were reduced to 0/1 values depending on if an event occured, we compared the Solar data to our DST data and created a confusion matrix to represent our results. We decided to calculate a couple other statistics as listed below, such as the correlation between the two arrays, the precision, the recall, and Heidke’s Skill Score. 
#
# The correlation explains how closely our an analyzed solar data follows the DST data, the precision calculates the amount of positive values that we predicted correctly, recall calculates, out of the amount of actual positives, how many events did we correctly predict, and Heidke’s Skill Score shows how much better our predictions are from random chance.

# %%
#Run the binary event analysis function on our dst event and solar wind event lists
print(binary_event_analysis(dst_binary, sw_binary))

# %% [markdown]
# **THIS SUCKS AND NEEDS TO BE EDITED**
#
# The range for Heidke’s Skill Score is from -1 to 1, with 0 representing random chance. Our values were a correlation of 0.4446, a precision of 0.5307, a recall of 0.5135, and a Hedike skill score of 0.4445. These values show that while our process of identifying these events using the solar data does work better than random chance, it is not necessarily the best. Similarly, our hit rate was 0.5135 and our false alarm rate was 0.0750, which shows our process is great at not predicting events when they did not happen, but not the best at predicting when they did. 
#
#
# ADD More anslysis about what the numbers mean    -- start with overall accuracy and then explain more about what each number represnts

# %% [markdown]
# ### Running Binary Event Analysis with Different Values
# To ensure that we picked the best window size and thresholds, we reran the binary event analysis with different values of threshold for identifying an event with solar wind and window size for the data. To evalute our success we looked at specifically the Heidke Skill score at it shows how well our prediction is at its job despite the large number of true numbers. We did this for window size from 4-7 days and 3 different cutoff thresholds for the solar wind (lower, medium and high thresholds). We only did a few because it would take 7+ hours to run all combinations. These threshold were chosen based on how the solar wind parameter were impacted for the 4/23/23 storm plotted above and the change in the new filtered data from the ifft. *Note - while this won't take 7 hours to run, it will take around 4-5 minutes to run so please be patient.*

# %%
# set for all due to theory of DST cutoff showing actual event
dstcutoff = -75 #nT
# for loop that gets skill score for 4-7 day window sizes and 3 different solar wind parameter thresholds for each
for window_size in range (4,8):
    # gets dst for the windowsize
    dst_binary = calc_dst_binary(window_size, cutoff)
    # low thresholds
    swcutoffs1 = [10, 15, 7.5, 5*10**5, 500]
    # medium thresholds
    swcutoffs2 = [15, 20, 10, 7.5*10**5, 550]
    # high thresholds
    swcutoffs3 = [20, 25, 12, 8*10**5, 600]
    sw_binary1 = calc_sw_binary(window_size, swcutoffs1)
    sw_binary2 = calc_sw_binary(window_size, swcutoffs2)
    sw_binary3 = calc_sw_binary(window_size, swcutoffs3)
    print(f'Window size: {window_size} days')
    print(f'\tHeidke Skill Score: {binary_event_analysis(dst_binary, sw_binary1, False)["heidke_skill_score"]:.6f} \t Overall Accuracy: {binary_event_analysis(dst_binary, sw_binary1, False)["proportion_correct"]:.6f}  for low solar wind thresholds')
    print(f'\tHeidke Skill Score: {binary_event_analysis(dst_binary, sw_binary2, False)["heidke_skill_score"]:.6f} \t Overall Accuracy: {binary_event_analysis(dst_binary, sw_binary2, False)["proportion_correct"]:.6f}  for day window and medium solar wind thresholds')
    print(f'\tHeidke Skill Score: {binary_event_analysis(dst_binary, sw_binary3, False)["heidke_skill_score"]:.6f} \t Overall Accuracy: {binary_event_analysis(dst_binary, sw_binary3, False)["proportion_correct"]:.6f}  for day window and high solar wind thresholds')


# %% [markdown]
# Overall, the medium thresholds produced the highest Heidke Skill score showing that the best thresholds for identifying solar events was our medium level thresholds. Additionally, as the window size increases the skill score increase. However, the 6 and 7 day window shows very little difference in the skill score. It is also important to note that the window size shouldn't be made too big because then it could overcorrect for the time delay/long term effect with the DST Index and therfore could be counting 2 seperate events as 1. Overall, the best overall accuracy does not align with the skill score. This is because a window size and/or thresholds can greatly increase the true negative number skewing the overall accuracy number to be higher despite the prediction being worse. This shows why we chose to focus on the skill score for our analysis as it removes the bias and high numbers of true negatives from the data and looks more at prediction accuracy.

# %% [markdown]
# # Data Analysis: ROC-Style Latin Hypercube Sampling
#
# Based on our above analysis, we decided to consider many of the factors that influenced how well our process identified events correctly. Since we were working with multiple values like window size, flow pressure, plasma temperature, and more, we wanted to find a way to alter the numerical values used for the thresholds of these values during analysis in order to see what was the best combination to predict these DST events. To streamline this process, we decided to use ROC-Style Latin Hypercube Sampling.
#
# ### What are LHS and ROC?
#
# Latin hypercube Sampling is a smart sampling technique to explore multidimensional parameter spaces. It takes each parameters range and ensures that each interval is sampled once. Since we would have had an output of thousands of points, using LHS allows us to cut this down to 200 points that is still representative of our data.    
#
# ROC curves are used to evaluate binary classification performance which shows the true positive rate vs the false positive rate. We chose to combine these two ideas and produce a plot that is representative of many possible combinations
#

# %% [markdown]
# ### Our Function and its Functionality
# *Note: takes a while to load*     
# The code explores a wide range of threshold combinations using Latin Hypercube Sampling, a method that ensures broad coverage of the parameter space with a limited number of trials. For each combination, the model classifies time periods as either containing a storm event or not, based on whether certain solar wind values exceed specified thresholds. These predictions are then compared against actual DST-based events to compute hit rates and false alarm rates. Ultimately, this analysis helps determine which parameter combinations are best at capturing true geomagnetic activity while minimizing false positives. The results are visualized in a ROC-style plot, with the top-performing configurations highlighted, providing insight into which solar wind features are most predictive of space weather disturbances.    

# %%
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
    '''
    DOC STRING
    '''
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
    '''
    DOC String
    '''
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
    'dst':      [-75, -150, -330],
    'B':        [5, 15, 30],
    'density':  [5, 15, 25],
    'pressure': [5, 10, 20],
    'temp':     [3e5, 7.5e5, 1e6],
    'velocity': [300, 550, 800]
}

#latin hypercube sampling -- parameter range is sampled evenly, but only 200 samples are taken
keys = list(param_lists.keys()) #hold parameters
#the seed argument keeps the numpy.random.Generator consistent when re-running the cell on the same machine
# results may vary across different computers
sampler = qmc.LatinHypercube(d=len(keys), seed=23) #do the latinhypercube
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

# print the top 3 combos
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
# ### ROC-Style LHS: Our Data's Outcome
#
# Our plot and results are shown above. For simplicity, the top three parameter combinations are labeled by the function. As you can see, any of these options was much better than those chosen by us in our original analysis, as seen by the hit rate vs false alarm rate. Of course there were several things that were not considered here, such as Heidke’s Skill Score, so in the future more analysis would be beneficial.  
#
# Latin Hypercube Sampling is particulary valuable for this analysis becayse it allowed us to examine all the parameters in our space much more effectively than a random sample. But, most importantly, it allowed us to get representative statistics without needed to comb through every single combination OR taking hours to run. One of the downsides, however, to Latin Hypercube Sampling is that all of the sampling outcomes are generated at once, meaning that each run produces a slightly different outcome. When we generated this plot multiple times, it was quite similar, but not exact. 
#

# %% [markdown]
# ## Conclusions and Next Steps
# Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# ## References
#

# %% [markdown]
#
# [1]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. Bowman, “The Heliospheric Current Sheet,” nasa.gov, Aug 5, 2013. [Online]. Available: https://www.nasa.gov/image-article/heliospheric-current-sheet/ [Accessed: April 14, 2025].
#
# [2]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. Case, J. Kasper, K. Korrek, and M. Stevens, “A Catalog of Solar Stream Interactions,” cfa.harvard.edu, Sept 17, 2021. [Online]. Available: https://www.cfa.harvard.edu/news/catalog-solar-stream-interactions/  [Accessed: April 14, 2025].
#
# [3]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D. Hathaway, “Coronal Features,” nasa.gov, Aug 11, 2014. [Online]. Available: https://solarscience.msfc.nasa.gov/feature3.shtml [Accessed: April 14, 2025].
#
# [4]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D. Hathaway, “The Solar Cycle,” *Living Rev. Sol. Phys.* 12, 4, 2015. Available: https://link.springer.com/content/pdf/10.1007/lrsp-2015-4.pdf [Accessed: April 14, 2025].
#
# [5]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I. Jolliffe, D. Stephenson, *Forecast Verification A Practitioner’s Guide in Atmospheric Science*, West Sussex, England:John Wiley &  Sons Ltd.
#
# [6]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;J. Borovsky and M. Denton, “The Difference Between CME-Driven Storms and CIR-Driven Storms,” *JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 111, A07S08*, July 26, 2006. Available: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2005JA011447 [Accessed: April 14, 2025].
#
# [7]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;J. Giacalone. Heliophysics Summer School. Powerpoint Lecture, Topic: “The Parker spiral magnetic field” University of Arizona, Lunar and Planetary Laboratory, Tuscan, AZ, Aug. 5 2022.
#
# [8]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;J. Palacios, A. Guerrero, C Cid, E. Saiz, and Y, Cerrato, “Defining Scale Thresholds for Geomagnetic Storms Through Statistics,” *Natural Hazards and Earth System Sciences Discussions*, April 12, 2015. Available: https://nhess.copernicus.org/preprints/nhess-2018-92/nhess-2018-92.pdf [Accessed: April 14, 2025].
#
# [9]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;K. Hansen, “Predicting Binary Events,” bar.rady.ucsd.edu, 2023. [Online]. Available: https://bar.rady.ucsd.edu/bin_class.html/ [Accessed: April 14, 2025].
#
# [10]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;M. Greenwood, “How to Use Latin Hypercube Sampling in Python,” hatchjs.com. [Online]. Available: https://hatchjs.com/latin-hypercube-sampling-python/ [Accessed: April 16, 2025].
#
# [11]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R. Schwenn, “Solar Wind Sources and Their Variations Over the Solar Cycle,” Space Sciences Series of ISSI, 2006. Available: https://link.springer.com/chapter/10.1007/978-0-387-69532-7_5#Bib1 [Accessed: April 14, 2025].
#
# [12]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SciPy community, “Latin Hypercube – SciPy v1.15.2 Manual,” docs.scipy.org. [Online]. Available: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html  [Accessed: April 16, 2025].
#
# [13]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Z. Bobbitt, “What is Latin Hypercube Sampling?,” statology.org, Sept 13, 2020. [Online]. Available: https://www.statology.org/latin-hypercube-sampling/ [Accessed: April 16, 2025].

# %% [markdown]
# ### Roles & Contributions

# %% [markdown]
#
