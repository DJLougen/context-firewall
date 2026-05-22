#!/usr/bin/env python3
"""
Generate visual outputs for documentation:
- Performance charts
- Compression comparison screenshots
- Architecture diagrams
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = project_root / 'docs' / 'images'
output_dir.mkdir(parents=True, exist_ok=True)


def plot_performance_comparison():
    """Generate performance comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    modes = ['Production\n(thread-safe)', 'High-Perf\n(no locks)', 'Rule-based\n(ML disabled)']
    throughput = [17000, 24000, 29000]
    colors = ['#2563eb', '#10b981', '#f59e0b']
    
    bars = ax.bar(modes, throughput, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,} msg/s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Throughput (messages/second)', fontsize=12, fontweight='bold')
    ax.set_title('Honey-Comb Performance Modes', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(throughput) * 1.15)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {output_dir / 'performance_comparison.png'}")


def plot_compression_ratio():
    """Generate compression ratio chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    content_types = ['Test Output\n(500 lines)', 'File Content\n(200 lines)', 
                     'Reasoning\n(15 lines)', 'Command Output\n(100 lines)',
                     'Error Trace\n(50 lines)']
    raw_tokens = [2500, 1200, 450, 800, 600]
    compressed_tokens = [30, 40, 120, 80, 50]
    
    x = np.arange(len(content_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, raw_tokens, width, label='Raw', 
                   color='#ef4444', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, compressed_tokens, width, label='Compressed', 
                   color='#10b981', edgecolor='black', linewidth=1.5)
    
    # Add compression ratio labels
    for i, (raw, comp) in enumerate(zip(raw_tokens, compressed_tokens)):
        ratio = raw / comp
        ax.text(i + width/2, comp + 50, f'{ratio:.1f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#10b981')
    
    ax.set_ylabel('Tokens', fontsize=12, fontweight='bold')
    ax.set_title('Compression Ratios by Content Type', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_ratios.png', bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {output_dir / 'compression_ratios.png'}")


def plot_latency_breakdown():
    """Generate latency breakdown chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data (in milliseconds)
    stages = ['Feature\nExtraction', 'Classification\n(Rules)', 'Classification\n(ML)', 
              'Compression', 'Total\n(Rules)', 'Total\n(ML)']
    latency = [0.01, 0.035, 0.8, 0.5, 0.545, 1.31]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899']
    
    bars = ax.barh(stages, latency, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, latency):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}ms',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Latency (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Breakdown by Stage', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_breakdown.png', bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {output_dir / 'latency_breakdown.png'}")


def plot_architecture_diagram():
    """Generate architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Honey-Comb Architecture', fontsize=16, fontweight='bold',
            ha='center', va='top')
    
    # Hot loop box
    hot_box = FancyBboxPatch((0.5, 5.5), 9, 3.5, boxstyle="round,pad=0.1",
                             facecolor='#dbeafe', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(hot_box)
    ax.text(1, 8.7, 'HOT LOOP (per message)', fontsize=11, fontweight='bold', color='#1e40af')
    ax.text(1, 8.3, '~0.035ms (rules) / ~0.8ms (ML)', fontsize=9, color='#1e40af')
    
    # Hot loop stages
    hot_stages = [
        (1, 7.5, 'Raw\nMessage', '#fef3c7'),
        (3, 7.5, 'Feature\nExtraction', '#dbeafe'),
        (5, 7.5, 'Classification', '#dbeafe'),
        (7, 7.5, 'Compression', '#dbeafe'),
        (9, 7.5, 'Compressed\nEntry', '#d1fae5'),
    ]
    
    for x, y, label, color in hot_stages:
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows between hot loop stages
    for i in range(len(hot_stages) - 1):
        x1, y1, _, _ = hot_stages[i]
        x2, y2, _, _ = hot_stages[i+1]
        arrow = FancyArrowPatch((x1+0.4, y1), (x2-0.4, y2),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='#2563eb')
        ax.add_patch(arrow)
    
    # Context window
    ctx_box = FancyBboxPatch((3.5, 4.5), 3, 0.6, boxstyle="round,pad=0.05",
                             facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(ctx_box)
    ax.text(5, 4.8, 'Context Window', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow from hot loop to context
    arrow = FancyArrowPatch((5, 5.5), (5, 5.1),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#f59e0b')
    ax.add_patch(arrow)
    
    # Cool loop box
    cool_box = FancyBboxPatch((0.5, 1), 9, 3, boxstyle="round,pad=0.1",
                              facecolor='#fce7f3', edgecolor='#ec4899', linewidth=2)
    ax.add_patch(cool_box)
    ax.text(1, 3.7, 'COOL LOOP (every N turns)', fontsize=11, fontweight='bold', color='#be185d')
    ax.text(1, 3.3, '~10-50ms', fontsize=9, color='#be185d')
    
    # Cool loop stages
    cool_stages = [
        (2.5, 2, 'Staleness\nDetection', '#fce7f3'),
        (5, 2, 'Budget\nEnforcement', '#fce7f3'),
        (7.5, 2, 'Clean\nContext', '#d1fae5'),
    ]
    
    for x, y, label, color in cool_stages:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows between cool loop stages
    for i in range(len(cool_stages) - 1):
        x1, y1, _, _ = cool_stages[i]
        x2, y2, _, _ = cool_stages[i+1]
        arrow = FancyArrowPatch((x1+0.6, y1), (x2-0.6, y2),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='#ec4899')
        ax.add_patch(arrow)
    
    # Arrow from context to cool loop
    arrow = FancyArrowPatch((5, 4.5), (5, 2.8),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#ec4899')
    ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture.png', bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {output_dir / 'architecture.png'}")


def main():
    """Generate all visual assets."""
    print("Generating visual assets for documentation...")
    print("=" * 60)
    
    plot_performance_comparison()
    plot_compression_ratio()
    plot_latency_breakdown()
    plot_architecture_diagram()
    
    print("=" * 60)
    print(f"All visuals saved to: {output_dir}")


if __name__ == "__main__":
    main()
