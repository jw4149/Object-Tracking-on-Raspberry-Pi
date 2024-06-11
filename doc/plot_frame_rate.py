import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Excel files
file_path_baseline = 'doc/frame_rate_log_baseline_1.xlsx'
file_path_ours = 'doc/frame_rate_log_ours_1.xlsx'
df_baseline = pd.read_excel(file_path_baseline)
df_ours = pd.read_excel(file_path_ours)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# Plot Frame Time
axes[0].plot(df_baseline['Frame Time (s)'], marker='x', linestyle='--', color='green', label='Baseline')
axes[0].plot(df_ours['Frame Time (s)'], marker='o', linestyle='-', color='orange', label='Ours')
# Draw average line and note the average value
axes[0].axhline(y=df_baseline['Frame Time (s)'].mean(), color='green', linestyle='--', label='Baseline Average')
axes[0].axhline(y=df_ours['Frame Time (s)'].mean(), color='orange', linestyle='--', label='Ours Average')
# Add text annotation for the average values
axes[0].text(0, df_baseline['Frame Time (s)'].mean(), f"Baseline Avg: {df_baseline['Frame Time (s)'].mean():.2f}", color='grey', ha='left', va='bottom', fontsize=18)
axes[0].text(0, df_ours['Frame Time (s)'].mean(), f"Ours Avg: {df_ours['Frame Time (s)'].mean():.2f}", color='grey', ha='left', va='bottom', fontsize=18)
axes[0].set_title('Frame Time Over Time')
axes[0].set_ylabel('Frame Time (s)')
axes[0].legend()
axes[0].grid(True)

# Plot Frame Rate
axes[1].plot(df_baseline['Frame Rate (fps)'], marker='x', linestyle='--', color='green', label='Baseline')
axes[1].plot(df_ours['Frame Rate (fps)'], marker='o', linestyle='-', color='orange', label='Ours')
# Draw average line and note the average value
axes[1].axhline(y=df_baseline['Frame Rate (fps)'].mean(), color='green', linestyle='--', label='Baseline Average')
axes[1].axhline(y=df_ours['Frame Rate (fps)'].mean(), color='orange', linestyle='--', label='Ours Average')
# Add text annotation for the average values on top of the line
axes[1].text(0.5, df_baseline['Frame Rate (fps)'].mean(), f"Baseline Avg: {df_baseline['Frame Rate (fps)'].mean():.2f}", color='grey', ha='left', va='bottom', fontsize=18)
axes[1].text(0.5, df_ours['Frame Rate (fps)'].mean(), f"Ours Avg: {df_ours['Frame Rate (fps)'].mean():.2f}", color='grey', ha='left', va='bottom', fontsize=18)

axes[1].set_title('Frame Rate Over Time')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Frame Rate (fps)')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('doc/frame_rate_comparison_plot.png')
plt.show()
plt.close()



# Descriptive Statistics
desc_baseline = df_baseline.describe()
desc_ours = df_ours.describe()
print("Baseline Descriptive Statistics:\n", desc_baseline)
print("\nOurs Descriptive Statistics:\n", desc_ours)

# Histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_baseline['Frame Time (s)'], kde=True, label='Baseline')
sns.histplot(df_ours['Frame Time (s)'], kde=True, label='Ours')
plt.legend()
plt.title('Frame Time Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df_baseline['Frame Rate (fps)'], kde=True, label='Baseline', color='orange')
sns.histplot(df_ours['Frame Rate (fps)'], kde=True, label='Ours', color='green')
plt.legend()
plt.title('Frame Rate Distribution')
plt.show()

# Box Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=[df_baseline['Frame Time (s)'], df_ours['Frame Time (s)']], palette='Set2')
plt.xticks([0, 1], ['Baseline', 'Ours'])
plt.title('Frame Time Comparison')

plt.subplot(1, 2, 2)
sns.boxplot(data=[df_baseline['Frame Rate (fps)'], df_ours['Frame Rate (fps)']], palette='Set2')
plt.xticks([0, 1], ['Baseline', 'Ours'])
plt.title('Frame Rate Comparison')
plt.show()

# T-tests
t_stat_time, p_value_time = stats.ttest_ind(df_baseline['Frame Time (s)'], df_ours['Frame Time (s)'])
t_stat_rate, p_value_rate = stats.ttest_ind(df_baseline['Frame Rate (fps)'], df_ours['Frame Rate (fps)'])
print(f"T-test for Frame Time: t-statistic = {t_stat_time}, p-value = {p_value_time}")
print(f"T-test for Frame Rate: t-statistic = {t_stat_rate}, p-value = {p_value_rate}")

# Cumulative Distribution Function (CDF)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.ecdfplot(df_baseline['Frame Time (s)'], label='Baseline')
sns.ecdfplot(df_ours['Frame Time (s)'], label='Ours')
plt.legend()
plt.title('CDF of Frame Time')

plt.subplot(1, 2, 2)
sns.ecdfplot(df_baseline['Frame Rate (fps)'], label='Baseline', color='orange')
sns.ecdfplot(df_ours['Frame Rate (fps)'], label='Ours', color='green')
plt.legend()
plt.title('CDF of Frame Rate')
plt.show()