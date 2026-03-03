import matplotlib.pyplot as plt


def plot_class_distribution(df, label_col="label", title="Class Distribution"):
    if df.empty or label_col not in df.columns:
        print("No data available for plotting.")
        return

    counts = df[label_col].value_counts()
    ax = counts.plot(kind="bar", figsize=(8, 4), title=title)
    ax.set_xlabel(label_col)
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()
