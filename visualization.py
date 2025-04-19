import matplotlib.pyplot as plt
import seaborn as sns

def plot_depth_vs_magnitude(df):
    plt.scatter(df['depth'], df['Magnitude'], s=10, alpha=0.7)
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.title("Depth vs. Magnitude")
    plt.show()
# Add other plot functions as needed
