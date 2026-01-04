"""Generate two separate clean panel figures for the evaluation protocol:
- panel_a: face splits (Train/Val/Test) with arrows and leakage note
- panel_b: forecasting timeline (Train/Val/Hold-out) with no-future-data arrow below

Run:
    python docs/figures/generate_eval_protocol_panels.py

Outputs:
    docs/figures/eval_protocol_panel_a.png
    docs/figures/eval_protocol_panel_b.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

out_dir = os.path.join(os.path.dirname(__file__))
os.makedirs(out_dir, exist_ok=True)

def make_panel_a(path, dpi=300):
    # Keep boxes unchanged; move external labels down and wrap text to avoid overlap
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    left = 0.04
    # adjust widths so the test (red) box can be square while fitting the figure
    h = 0.28
    val_w = 0.12
    test_w = val_w
    train_w = 1.0 - left - val_w - test_w
    y = 0.55

    ax.add_patch(patches.Rectangle((left, y), train_w, h, facecolor='#a6cee3', edgecolor='k'))
    ax.add_patch(patches.Rectangle((left+train_w, y), val_w, h, facecolor='#b2df8a', edgecolor='k'))
    ax.add_patch(patches.Rectangle((left+train_w+val_w, y), test_w, h, facecolor='#fb9a99', edgecolor='k'))

    ax.text(left + train_w/2, y+h/2, 'Train\n70%', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(left + train_w + val_w/2, y+h/2, 'Val\n15%', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(left + train_w + val_w + test_w/2, y+h/2, 'Test\n15%', ha='center', va='center', fontsize=12, weight='bold')

    # arrows and labels
    # Position arrows pointing down from box centers, then stack external labels vertically
    arrow_y = y - 0.02
    label_y_base = y - 0.18
    wrap_width = 24

    # Train label (top)
    tx = left + train_w * 0.35
    ax.annotate('', xy=(tx, arrow_y+0.01), xytext=(tx, y-0.005), arrowprops=dict(arrowstyle='->', lw=1.2))
    train_label = textwrap.fill('Train recognition / landmarks / reconstruction', wrap_width)
    ax.text(tx, label_y_base, train_label, ha='center', fontsize=9)

    # Val label (middle)
    vx = left + train_w + val_w * 0.5
    ax.annotate('', xy=(vx, arrow_y+0.01), xytext=(vx, y-0.005), arrowprops=dict(arrowstyle='->', lw=1.2))
    val_label = textwrap.fill('Hyperparam & threshold selection', wrap_width)
    ax.text(vx, label_y_base - 0.06, val_label, ha='center', fontsize=9)

    # Test label (bottom)
    tx2 = left + train_w + val_w + test_w * 0.5
    ax.annotate('', xy=(tx2, arrow_y+0.01), xytext=(tx2, y-0.005), arrowprops=dict(arrowstyle='->', lw=1.2))
    test_label = textwrap.fill('Final metrics: accuracy, F1, ROC, NME, vertex error', wrap_width)
    ax.text(tx2, label_y_base - 0.18, test_label, ha='center', fontsize=9)

    ax.text(0.5, 0.94, 'No identity appears in more than one split', ha='center', fontsize=11, weight='bold')
    ax.set_title('(a) Face model evaluation protocol', loc='left', fontsize=12, weight='bold')

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def make_panel_b(path, dpi=300):
    # Keep boxes unchanged; move the long arrow and its label below the timeline with spacing
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    train_x = 0.03
    gap = 0.04
    # make hold-out square using same height as box; adjust train/val to fit
    h = 0.34
    val_w = 0.11
    hold_w = val_w
    train_w = 1.0 - train_x - gap - val_w - gap - hold_w
    y = 0.5

    ax.add_patch(patches.Rectangle((train_x, y), train_w, h, facecolor='#a6cee3', edgecolor='k'))
    ax.add_patch(patches.Rectangle((train_x + train_w + gap, y), val_w, h, facecolor='#b2df8a', edgecolor='k'))
    ax.add_patch(patches.Rectangle((train_x + train_w + gap + val_w + gap, y), hold_w, h, facecolor='#fb9a99', edgecolor='k', linewidth=1.8))

    # Short labels inside boxes and explanatory text below each box to avoid overlap
    ax.text(train_x + train_w/2, y+h/2, 'Train', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(train_x + train_w + gap + val_w/2, y+h/2, 'Val', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(train_x + train_w + gap + val_w + gap + hold_w/2, y+h/2, 'Hold-out', ha='center', va='center', fontsize=10, weight='bold')

    # Explanatory labels below boxes (wrapped)
    label_y = y - 0.18
    wrap_w = 30
    train_desc = textwrap.fill('Train period (months ...)', wrap_w)
    val_desc = textwrap.fill('Validation period (hyperparam tuning)', wrap_w)
    hold_desc = textwrap.fill('Hold-out period (evaluation only)', wrap_w)

    ax.text(train_x + train_w/2, label_y, train_desc, ha='center', fontsize=10)
    ax.text(train_x + train_w + gap + val_w/2, label_y, val_desc, ha='center', fontsize=10)
    ax.text(train_x + train_w + gap + val_w + gap + hold_w/2, label_y - 0.08, hold_desc, ha='center', fontsize=10)

    # long arrow below timeline, placed further down to avoid overlap with boxes
    arrow_y = label_y - 0.22
    start = train_x
    end = train_x + train_w + gap + val_w + gap + hold_w
    ax.annotate('', xy=(start, arrow_y), xytext=(end, arrow_y), arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text((start+end)/2, arrow_y - 0.06, 'No future data used for training', ha='center', fontsize=11, style='italic')

    ax.set_title('(b) Crime forecasting temporal split', loc='left', fontsize=12, weight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    a_path = os.path.join(out_dir, 'eval_protocol_panel_a.png')
    b_path = os.path.join(out_dir, 'eval_protocol_panel_b.png')
    make_panel_a(a_path)
    make_panel_b(b_path)
    print('Saved panels:', a_path, b_path)
