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
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

tconvert = lambda x: dt.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S.%fZ')

data = np.genfromtxt('OMNI2_H0_MRG1HR_2201110.csv', names=True, delimiter=',', skip_header=78, encoding='utf-8',converters={0:tconvert}, dtype=None)

time = data['TIME_AT_CENTER_OF_HOUR_yyyymmddThhmmsssssZ']
swavgB = data['1AU_IP_MAG_AVG_B_nT']
swvelocity = data['1AU_IP_PLASMA_SPEED_Kms']
swpressure = data['1AU_IP_FLOW_PRESSURE_nPa']
swtemp = data ['1AU_IP_PLASMA_TEMP_Deg_K']
swdensity = data['1AU_IP_N_ION_Per_cc']
dst = data['1H_DST_nT']

# %%
arrays = [swavgB, swdensity, swvelocity, swpressure, swtemp]
filters = [80, 100, 1200, 60, 1*10**7]

for array, threshold in zip(arrays, filters):
    for index, value in enumerate(array):
        if value >= threshold:
            array[index] = np.nan

# %%
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

# %%
data = {'Average Magnetic Field at 1 AU|Magnetic Field (nT)': (time,swavgB),
        'Solar Wind Velocity|Velocity (km/s)': (time,swvelocity),
        'Plasma Flow Pressure|Pressure (nPa)': (time,swpressure),
        'Plasma Temperature|Temperature (K)': (time,swtemp),
        'Ion Number Density|Density (per cc)': (time,swdensity),
        'DST Index|DST (nT)': (time,dst)}

fig, axes = plt.subplots(6,1, figsize = (12,20))
fig.suptitle('May 2024 Storm')
# for loop to add data to each plot
for ax, (label, (x, y)) in zip(axes.flat, data.items()):
    #add data to the plot
    ax.set_xlim(dt.datetime(2024,5,10),dt.datetime(2024,5,15))
    ax.plot(x,y)
    # adds proper titles and labels
    title, space, ytext = label.partition('|')
    ax.set_title(title)   
    ax.set_xlabel('Date')
    ax.set_ylabel(ytext)
fig.tight_layout()

# %% [markdown]
# ### Question 1
# Write a function to read the *.csv files using numpy.genfromtxt. Leverage the example above to ensure success.
#
#

# %% [markdown]
# ### Question 2
# Description of what you need to do and interpretation of results (if applicable)
#
#

# %% [markdown]
# ### Question 3
# Description of what you need to do and interpretation of results (if applicable)
#

# %% [markdown]
# ## Conclusions
# Synthesize the conclusions from your results section here. Give overarching conclusions. Tell us what you learned.
# ## References
# List any references used
