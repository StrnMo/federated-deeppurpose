"""
Generate all visualization plots for the federated learning project.
Run this script after centralized and federated training are complete.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

def plot_centralized_loss():
    """Plot centralized training loss curve."""
    # Data from your centralized training
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_loss = [0.571, 0.585, 1.035, 1.102, 1.010, 0.619, 1.086, 0.733, 1.085, 1.006]
    val_loss = [0.531, 1.172, 0.768, 0.513, 1.187, 0.625, 0.998, 1.069, 0.544, 0.558]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Centralized Model Training (10 Epochs)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/centralized_loss_curve.png', dpi=150)
    plt.savefig('results/centralized_loss_curve.pdf')  # PDF for papers
    plt.close()
    print("✅ Saved: results/centralized_loss_curve.png")

def plot_federated_comparison():
    """Plot bar chart comparing federated clients to centralized baseline."""
    clients = ['Client 0', 'Client 1', 'Client 2', 'Client 3', 'Client 4', 'Centralized']
    c_index = [0.630, 0.693, 0.596, 0.669, 0.698, 0.731]
    colors = ['#1f77b4'] * 5 + ['#d62728']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(clients, c_index, color=colors, edgecolor='black', linewidth=1)
    plt.axhline(y=0.731, color='#d62728', linestyle='--', linewidth=2, 
                label='Centralized Baseline (0.731)')
    plt.ylabel('Concordance Index', fontsize=12)
    plt.title('Federated Client Performance vs Centralized Baseline', fontsize=14)
    plt.ylim(0.55, 0.75)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, c_index):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/federated_comparison.png', dpi=150)
    plt.savefig('results/federated_comparison.pdf')
    plt.close()
    print("✅ Saved: results/federated_comparison.png")

def plot_client_distribution():
    """Plot client data distribution (samples and unique drugs)."""
    clients = ['Client 0', 'Client 1', 'Client 2', 'Client 3', 'Client 4']
    samples = [5746, 5746, 5746, 5746, 7072]
    unique_drugs = [13, 13, 13, 13, 16]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Samples bar chart
    bars1 = ax1.bar(clients, samples, color='steelblue', edgecolor='black')
    ax1.set_ylabel('Number of Drug-Target Pairs', fontsize=12)
    ax1.set_title('Samples per Client', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, samples):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val}', ha='center', va='bottom', fontsize=10)
    
    # Unique drugs bar chart
    bars2 = ax2.bar(clients, unique_drugs, color='coral', edgecolor='black')
    ax2.set_ylabel('Number of Unique Drugs', fontsize=12)
    ax2.set_title('Unique Drugs per Client', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, unique_drugs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/client_data_distribution.png', dpi=150)
    plt.savefig('results/client_data_distribution.pdf')
    plt.close()
    print("✅ Saved: results/client_data_distribution.png")

def plot_performance_table():
    """Create a summary table of all results."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Model', 'MSE ↓', 'Pearson ↑', 'C-Index ↑'],
        ['Centralized (10 epochs)', '0.615', '0.440', '0.731'],
        ['Client 0 (5 epochs)', '0.822', '0.212', '0.630'],
        ['Client 1 (5 epochs)', '0.774', '0.365', '0.693'],
        ['Client 2 (5 epochs)', '0.474', '0.200', '0.596'],
        ['Client 3 (5 epochs)', '0.721', '0.342', '0.669'],
        ['Client 4 (5 epochs)', '0.894', '0.369', '0.698'],
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.25, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Highlight header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight centralized row
    for i in range(4):
        table[(1, i)].set_facecolor('#E6F0FA')
    
    plt.title('Federated Learning Results Summary', fontsize=14, pad=20, fontweight='bold')
    plt.savefig('results/performance_table.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/performance_table.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Saved: results/performance_table.png")

def plot_combined_summary():
    """Create a single combined figure with all plots."""
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Centralized loss curve
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_loss = [0.571, 0.585, 1.035, 1.102, 1.010, 0.619, 1.086, 0.733, 1.085, 1.006]
    val_loss = [0.531, 1.172, 0.768, 0.513, 1.187, 0.625, 0.998, 1.069, 0.544, 0.558]
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Centralized Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Federated comparison
    ax2 = fig.add_subplot(2, 2, 2)
    clients = ['C0', 'C1', 'C2', 'C3', 'C4', 'Central']
    c_index = [0.630, 0.693, 0.596, 0.669, 0.698, 0.731]
    colors = ['#1f77b4'] * 5 + ['#d62728']
    bars = ax2.bar(clients, c_index, color=colors)
    ax2.axhline(y=0.731, color='#d62728', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Concordance Index')
    ax2.set_title('Federated vs Centralized')
    ax2.set_ylim(0.55, 0.75)
    for bar, val in zip(bars, c_index):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Client distribution (samples)
    ax3 = fig.add_subplot(2, 2, 3)
    samples = [5746, 5746, 5746, 5746, 7072]
    ax3.bar(clients[:5], samples, color='steelblue')
    ax3.set_ylabel('Samples')
    ax3.set_title('Data per Client')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Client distribution (unique drugs)
    ax4 = fig.add_subplot(2, 2, 4)
    unique_drugs = [13, 13, 13, 13, 16]
    ax4.bar(clients[:5], unique_drugs, color='coral')
    ax4.set_ylabel('Unique Drugs')
    ax4.set_title('Unique Drugs per Client')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Federated Learning for Drug-Target Interaction Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/combined_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/combined_summary.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Saved: results/combined_summary.png")

if __name__ == "__main__":
    print("=" * 50)
    print("Generating Visualization Plots")
    print("=" * 50)
    plot_centralized_loss()
    plot_federated_comparison()
    plot_client_distribution()
    plot_performance_table()
    plot_combined_summary()
    print("\n" + "=" * 50)
    print("All plots saved to 'results/' folder")
    print("Files created:")
    print("  - centralized_loss_curve.png/.pdf")
    print("  - federated_comparison.png/.pdf")
    print("  - client_data_distribution.png/.pdf")
    print("  - performance_table.png/.pdf")
    print("  - combined_summary.png/.pdf")
    print("=" * 50)