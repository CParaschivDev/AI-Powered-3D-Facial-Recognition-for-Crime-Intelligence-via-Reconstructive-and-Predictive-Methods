"""
Generate Crime Data ETL and Forecasting Pipeline Diagram

This script creates a visual diagram showing:
- UK Police CSVs → Pandas aggregation → crime_full.parquet → Polars service → FastAPI → Dashboard
- Forecasting branch: crime_full.parquet → Prophet → predictions table → Analytics API
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_crime_pipeline_diagram(output_path='reports/crime_etl_pipeline.png'):
    """Generate the Crime Data ETL and Forecasting Pipeline diagram"""
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'ingestion': '#e1f5ff',
        'storage': '#fff4e6',
        'api': '#e7f5e7',
        'frontend': '#f3e5f5',
        'forecast': '#ffe6e6'
    }
    
    def draw_box(x, y, width, height, text, color, fontsize=11, bold=False):
        """Draw a rounded box with text"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor='#333',
            linewidth=3
        )
        ax.add_patch(box)
        
        weight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text, 
               ha='center', va='center', fontsize=fontsize,
               weight=weight, family='sans-serif',
               linespacing=1.8)
    
    def draw_arrow(x1, y1, x2, y2, style='solid', color='black', width=3):
        """Draw an arrow between two points"""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            linestyle=style,
            color=color, 
            linewidth=width,
            mutation_scale=30
        )
        ax.add_patch(arrow)
    
    # Title
    ax.text(9, 13.2, 'Crime Data ETL and Forecasting Pipeline', 
           ha='center', va='top', fontsize=24, weight='bold')
    
    # ===== LEFT COLUMN: MAIN PIPELINE =====
    
    # 1. Data Source
    draw_box(3, 11.5, 5, 1, 'UK Police CSV Files\n(monthly CSVs per force)', 
            colors['ingestion'], 12, bold=True)
    
    draw_arrow(5.5, 11.5, 5.5, 10.3)
    
    # 2. ETL Processing
    draw_box(2.5, 9, 6, 1.3, 
            'Pandas Aggregation (predict.py)\n\n' +
            '• Load monthly CSVs\n• Aggregate to national totals\n• Distribute to daily granularity',
            colors['ingestion'], 10)
    
    draw_arrow(5.5, 9, 5.5, 7.8)
    
    # 3. Storage
    draw_box(3, 6.8, 5, 1, 'crime_full.parquet\n(columnar format)', 
            colors['storage'], 12, bold=True)
    
    draw_arrow(5.5, 6.8, 5.5, 5.6)
    
    # 4. Service Layer
    draw_box(2.5, 4.2, 6, 1.4,
            'Polars Service (crime_service.py)\n\n' +
            '• get_crime_dataframe()\n• LRU caching\n• Lazy evaluation',
            colors['api'], 10)
    
    draw_arrow(5.5, 4.2, 5.5, 3)
    
    # 5. API Layer
    draw_box(2.5, 1.5, 6, 1.5,
            'Crime Routes\n(backend/api/routes/crime.py)\n\n' +
            '/crime/forces/monthly\n/crime/hotspots/latest\n/crime/trends',
            colors['api'], 10)
    
    draw_arrow(5.5, 1.5, 5.5, 0.3)
    
    # ===== RIGHT COLUMN: FORECASTING BRANCH =====
    
    # Branch arrow from Parquet to Prophet
    draw_arrow(8, 7.3, 10, 9, style='dashed', color='#d32f2f', width=3.5)
    
    # Prophet
    draw_box(10, 9, 6, 1.3,
            'Prophet Model (predict.py)\n\n' +
            '• Time series forecasting\n• Trend decomposition\n• Seasonality detection',
            colors['forecast'], 10)
    
    draw_arrow(13, 9, 13, 7.8)
    
    # Predictions DB
    draw_box(10.5, 6.8, 5, 1,
            'Predictions Table\n(SQLite/PostgreSQL)',
            colors['forecast'], 12, bold=True)
    
    draw_arrow(13, 6.8, 13, 5.6)
    
    # Analytics API
    draw_box(10, 4.2, 6, 1.4,
            'Analytics API\n/api/v1/analytics/predictions\n\n' +
            '• National forecasts (30-365 days)\n• Daily crime predictions',
            colors['forecast'], 10)
    
    draw_arrow(13, 4.2, 9, 0.3, style='dashed', color='#d32f2f', width=3.5)
    
    # ===== BOTTOM: DASHBOARD =====
    
    draw_box(4.5, 0, 9, 0.3,
            'Dashboard Charts  •  Heatmaps  •  Temporal Patterns  •  Live Alerts  •  Crime Forecasts',
            colors['frontend'], 10, bold=True)
    
    # ===== LEGEND =====
    
    ax.text(2, -1.2, 'Legend:', fontsize=12, weight='bold')
    
    draw_arrow(2, -1.5, 3.5, -1.5, style='solid', color='black', width=3)
    ax.text(3.8, -1.5, 'Main ETL Pipeline', ha='left', va='center', fontsize=10)
    
    draw_arrow(8, -1.5, 9.5, -1.5, style='dashed', color='#d32f2f', width=3.5)
    ax.text(9.8, -1.5, 'Forecasting Branch', ha='left', va='center', fontsize=10)
    
    # Footer
    ax.text(9, -2.2, 
           'Complete data flow from raw UK Police CSV files through analytics services to dashboard visualization and predictive forecasting',
           ha='center', fontsize=10, style='italic', color='#666')
    
    # Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Crime pipeline diagram saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Crime pipeline diagram saved to: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    create_crime_pipeline_diagram()
    create_crime_pipeline_diagram('docs/figures/crime_etl_pipeline.png')

