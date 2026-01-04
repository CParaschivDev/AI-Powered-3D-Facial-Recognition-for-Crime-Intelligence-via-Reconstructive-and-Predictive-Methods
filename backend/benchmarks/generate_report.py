import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
CSV_PATH = os.path.join(REPORTS_DIR, "benchmark_results.csv")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "benchmark_report.png")

def generate_report():
    """Loads benchmark data and generates a PNG report."""
    if not os.path.exists(CSV_PATH):
        logger.error(f"Benchmark results not found at {CSV_PATH}. Please run 'run_benchmarks.py' first.")
        return

    logger.info(f"Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    sns.set_theme(style="whitegrid")
    
    # Create a 2x2 grid for plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('System Performance Benchmarks', fontsize=20, weight='bold')

    # 1. Runtime Plot by Quality
    ax1 = axes[0, 0]
    sns.barplot(data=df, x='quality', y='runtime_sec', hue='device', ax=ax1)
    ax1.set_title('Runtime by Image Quality and Device', fontsize=14, weight='bold')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_xlabel('Image Quality')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Memory Usage Plot
    ax2 = axes[0, 1]
    sns.barplot(data=df, x='task', y='peak_mem_mb', hue='device', ax=ax2)
    ax2.set_title('Peak Memory Usage by Task and Device', fontsize=14, weight='bold')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_xlabel('Task')

    # 3. Accuracy Plot (only for recognition)
    ax3 = axes[1, 0]
    rec_df = df[df['task'] == 'recognition'].dropna(subset=['accuracy'])
    if not rec_df.empty:
        sns.barplot(data=rec_df, x='quality', y='accuracy', hue='device', ax=ax3)
        ax3.set_title('Recognition Accuracy by Image Quality', fontsize=14, weight='bold')
        ax3.set_ylabel('Accuracy (Mock)')
        ax3.set_xlabel('Image Quality')
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center')
        ax3.set_title('Recognition Accuracy', fontsize=14, weight='bold')


    # 4. Runtime by Task
    ax4 = axes[1, 1]
    sns.barplot(data=df, x='task', y='runtime_sec', hue='device', ax=ax4)
    ax4.set_title('Runtime by Task and Device', fontsize=14, weight='bold')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_xlabel('Task')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    logger.info(f"Saving report to {OUTPUT_PATH}")
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    logger.info("Report generation complete.")

if __name__ == "__main__":
    generate_report()
