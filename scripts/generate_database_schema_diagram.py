"""
Generate Crime Intelligence Database Schema Diagram - Clean and Simple

Shows the key tables and their relationships with encrypted fields marked.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def create_database_schema_diagram(output_path='reports/database_schema.png'):
    """Generate the Crime Intelligence Database Schema diagram"""
    
    fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define solid colors for legend
    pk_color = '#FFEB3B'      # Bright Yellow
    fk_color = '#E1BEE7'      # Light Purple
    encrypted_color = '#FFCDD2'  # Light Pink
    
    def draw_table(x, y, table_name, fields):
        """Draw a simple database table"""
        width = 3.5
        row_height = 0.4
        header_height = 0.6
        total_height = header_height + (len(fields) * row_height)

        # Header
        header = Rectangle((x, y + total_height - header_height), width, header_height,
                          facecolor='#1976D2', edgecolor='black', linewidth=2, zorder=3)
        ax.add_patch(header)
        ax.text(x + width/2, y + total_height - header_height/2, table_name,
               ha='center', va='center', fontsize=11, weight='bold', color='white', zorder=4)

        # Body
        body = Rectangle((x, y), width, total_height - header_height,
                         facecolor='#E3F2FD', edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(body)

        # Fields
        field_positions = {}
        for i, field in enumerate(fields):
            field_y = y + total_height - header_height - (i + 0.5) * row_height
            field_positions[field] = (x + 0.15, field_y)
            # Determine field type and color
            bg_color = None
            if '(PK)' in field:
                bg_color = pk_color
            elif '(FK)' in field:
                bg_color = fk_color
            elif '[ENC]' in field:
                bg_color = encrypted_color
            # Draw background if needed
            if bg_color:
                ax.add_patch(Rectangle((x + 0.05, field_y - row_height/2 + 0.05),
                                       width - 0.1, row_height - 0.1,
                                       facecolor=bg_color, edgecolor='none', alpha=0.8, zorder=3))
            # Draw field text
            ax.text(x + 0.15, field_y, field, ha='left', va='center',
                    fontsize=9, weight='bold' if '(PK)' in field else 'normal', zorder=4)
        return total_height, field_positions
    
    # Title
    ax.text(10, 13, 'Crime Intelligence Database Schema', 
           ha='center', fontsize=20, weight='bold')
    
    # ===== ROW 1: Core Tables =====
    y1 = 9.5
    user_h, user_fields = draw_table(0.5, y1, 'User', [
        'id (PK)',
        'username',
        'hashed_password [ENC]',
        'role',
        'created_at'
    ])
    evidence_h, evidence_fields = draw_table(5, y1, 'Evidence', [
        'id (PK)',
        'user_id (FK)',
        'evidence_type',
        'file_path [ENC]',
        'uploaded_at'
    ])
    identities_h, identities_fields = draw_table(9.5, y1, 'Identities', [
        'id (PK)',
        'name [ENC]',
        'alias [ENC]',
        'national_id [ENC]',
        'risk_level'
    ])
    prediction_h, prediction_fields = draw_table(14, y1, 'Prediction', [
        'id (PK)',
        'timestamp',
        'location',
        'crime_type',
        'predicted_count'
    ])
    # ===== ROW 2: Processing Tables =====
    y2 = 5.5
    snapshot_h, snapshot_fields = draw_table(0.5, y2, 'Snapshot', [
        'id (PK)',
        'evidence_id (FK)',
        'user_id (FK)',
        'reconstruction_obj [ENC]',
        'landmarks [ENC]',
        'face_image [ENC]',
        'created_at'
    ])
    identityembedding_h, identityembedding_fields = draw_table(5, y2, 'IdentityEmbedding', [
        'id (PK)',
        'identity_id (FK)',
        'snapshot_id (FK)',
        'embedding [ENC]',
        'model_version',
        'confidence',
        'extracted_at'
    ])
    modelversion_h, modelversion_fields = draw_table(9.5, y2, 'ModelVersion', [
        'id (PK)',
        'model_name',
        'version',
        'framework',
        'accuracy',
        'training_date',
        'is_active'
    ])
    biasmetrics_h, biasmetrics_fields = draw_table(14, y2, 'BiasMetrics', [
        'id (PK)',
        'model_version_id (FK)',
        'metric_type',
        'demographic_group',
        'score',
        'threshold'
    ])
    # ===== ROW 3: Audit =====
    y3 = 2
    auditlog_h, auditlog_fields = draw_table(0.5, y3, 'AuditLog', [
        'id (PK)',
        'user_id (FK)',
        'action',
        'table_name',
        'record_id',
        'changes [ENC]',
        'timestamp'
    ])
    
    # ===== RELATIONSHIPS =====
    def draw_relation(x1, y1, x2, y2, label='1:N'):
        """Draw a professional relationship line with arrow"""
        # compute gentle curvature based on direction
        dx = x2 - x1
        rad = 0.12 if dx >= 0 else -0.12
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=1.0, color='#666', alpha=0.5,
                     connectionstyle=f"arc3,rad={rad}"), zorder=1)
        # relationship label intentionally omitted (drawn on arrows in legend only)
    
    # User (0.5 to 4) -> Evidence (5 to 8.5) - horizontal
    draw_relation(4, 10.5, 5, 10.5)
    
    # User (center 2.25) -> Snapshot (center 2.25) - vertical
    draw_relation(2.25, 9.5, 2.25, 8.5)
    
    # Evidence (center 6.75, bottom 9.5) -> Snapshot (right edge 4, top 8.5)
    draw_relation(6.75, 9.5, 4, 8.3)
    
    # Snapshot (4) -> IdentityEmbedding (5) - horizontal
    draw_relation(4, 7, 5, 7)
    
    # Identities (left 9.5) -> IdentityEmbedding (right 8.5)
    draw_relation(9.5, 9.2, 8.5, 8.3)
    
    # Identities (13) -> Prediction (14) - horizontal
    draw_relation(13, 10.5, 14, 10.5)
    
    # ===== RELATIONSHIPS (field-to-field) =====
    def draw_relation(fields_from, field_from, fields_to, field_to, label='1:N'):
        """Draw a professional relationship line from FK to PK field"""
        # Get coordinates from field positions
        x1, y1 = fields_from[field_from]
        x2, y2 = fields_to[field_to]
        # Add margin so arrows/labels don't overlap field text
        margin = 0.25
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            length = 1
        x1m = x1 + margin * dx / length
        y1m = y1 + margin * dy / length
        x2m = x2 - margin * dx / length
        y2m = y2 - margin * dy / length
        # Use a gentler curved connection style, arrows drawn beneath tables
        rad_local = 0.12 if (x2m - x1m) >= 0 else -0.12
        ax.annotate('', xy=(x2m, y2m), xytext=(x1m, y1m),
                   arrowprops=dict(arrowstyle='->', lw=1.0, color='#666', alpha=0.5,
                                 connectionstyle=f"arc3,rad={rad_local}"), zorder=1)
        # relationship label intentionally omitted for field-to-field arrows
    # User.id (PK) -> Evidence.user_id (FK)
    draw_relation(user_fields, 'id (PK)', evidence_fields, 'user_id (FK)')
    # User.id (PK) -> Snapshot.user_id (FK)
    draw_relation(user_fields, 'id (PK)', snapshot_fields, 'user_id (FK)')
    # Evidence.id (PK) -> Snapshot.evidence_id (FK)
    draw_relation(evidence_fields, 'id (PK)', snapshot_fields, 'evidence_id (FK)')
    # Snapshot.id (PK) -> IdentityEmbedding.snapshot_id (FK)
    draw_relation(snapshot_fields, 'id (PK)', identityembedding_fields, 'snapshot_id (FK)')
    # Identities.id (PK) -> IdentityEmbedding.identity_id (FK)
    draw_relation(identities_fields, 'id (PK)', identityembedding_fields, 'identity_id (FK)')
    # ModelVersion.id (PK) -> BiasMetrics.model_version_id (FK)
    draw_relation(modelversion_fields, 'id (PK)', biasmetrics_fields, 'model_version_id (FK)')
    # User.id (PK) -> AuditLog.user_id (FK)
    draw_relation(user_fields, 'id (PK)', auditlog_fields, 'user_id (FK)')
    legend_y = 0.5
    legend_x = 5
    box_size = 0.5
    spacing = 3.5
    
    # Primary Key
    ax.add_patch(Rectangle((legend_x, legend_y), box_size, box_size,
                          facecolor=pk_color, edgecolor='black', linewidth=2))
    ax.text(legend_x + box_size + 0.2, legend_y + box_size/2, 'Primary Key (PK)',
           ha='left', va='center', fontsize=10, weight='bold')
    
    # Foreign Key
    ax.add_patch(Rectangle((legend_x + spacing, legend_y), box_size, box_size,
                          facecolor=fk_color, edgecolor='black', linewidth=2))
    ax.text(legend_x + spacing + box_size + 0.2, legend_y + box_size/2, 'Foreign Key (FK)',
           ha='left', va='center', fontsize=10, weight='bold')
    
    # Encrypted Field
    ax.add_patch(Rectangle((legend_x + 2*spacing, legend_y), box_size, box_size,
                          facecolor=encrypted_color, edgecolor='black', linewidth=2))
    ax.text(legend_x + 2*spacing + box_size + 0.2, legend_y + box_size/2, 'Encrypted [ENC]',
           ha='left', va='center', fontsize=10, weight='bold')
    
    # Relationship line in legend
    ax.plot([legend_x + 3*spacing, legend_x + 3*spacing + box_size + 0.3],
           [legend_y + box_size/2, legend_y + box_size/2],
           'k-', linewidth=2, alpha=0.6)
    # Single-word label to the right of the legend line
    ax.text(legend_x + 3*spacing + box_size + 0.5, legend_y + box_size/2, 'Relation',
           ha='left', va='center', fontsize=10, weight='bold')
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Database schema diagram saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Database schema diagram saved to: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    create_database_schema_diagram()
    create_database_schema_diagram('docs/figures/database_schema.png')
