"""Generate a two-panel figure summarising evaluation protocols:
- (a) Face model identity-level split (Train/Val/Test) and data-flow
- (b) Crime forecasting temporal split (Train / Validation / Hold-out)

Run:
    python docs/figures/generate_eval_protocol_figure.py

Outputs:
    docs/figures/eval_protocol.png
    docs/figures/eval_protocol.svg
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

out_dir = os.path.join(os.path.dirname(__file__))
os.makedirs(out_dir, exist_ok=True)

png_path = os.path.join(out_dir, 'eval_protocol.png')
svg_path = os.path.join(out_dir, 'eval_protocol.svg')

fig = plt.figure(figsize=(14,6), constrained_layout=True)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# Panel 1: Face model evaluation protocol (improved spacing)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.axis('off')

# Draw split bar with more margin
train_w = 0.68
val_w = 0.16
test_w = 0.16
h = 0.18
y = 0.6
left_margin = 0.04
ax1.add_patch(patches.Rectangle((left_margin, y), train_w, h, facecolor='#a6cee3', edgecolor='k'))
ax1.add_patch(patches.Rectangle((left_margin+train_w, y), val_w, h, facecolor='#b2df8a', edgecolor='k'))
ax1.add_patch(patches.Rectangle((left_margin+train_w+val_w, y), test_w, h, facecolor='#fb9a99', edgecolor='k'))

ax1.text(left_margin + train_w/2, y+h/2, 'Train\n70%', va='center', ha='center', fontsize=10, weight='bold')
ax1.text(left_margin + train_w + val_w/2, y+h/2, 'Val\n15%', va='center', ha='center', fontsize=10, weight='bold')
ax1.text(left_margin + train_w + val_w + test_w/2, y+h/2, 'Test\n15%', va='center', ha='center', fontsize=10, weight='bold')

# Arrows to boxes (positioned to avoid overlap)
arrow_y = y - 0.08
ax1.annotate('', xy=(left_margin + train_w*0.35, arrow_y+0.02), xytext=(left_margin + train_w*0.35, y-0.005), arrowprops=dict(arrowstyle='->'))
ax1.text(left_margin + train_w*0.35, arrow_y - 0.04, 'Train recognition / landmarks / reconstruction', ha='center', fontsize=9)

ax1.annotate('', xy=(left_margin + train_w + val_w*0.5, arrow_y+0.02), xytext=(left_margin + train_w + val_w*0.5, y-0.005), arrowprops=dict(arrowstyle='->'))
ax1.text(left_margin + train_w + val_w*0.5, arrow_y - 0.04, 'Hyper-parameter &\nthreshold selection', ha='center', fontsize=9)

ax1.annotate('', xy=(left_margin + train_w + val_w + test_w*0.5, arrow_y+0.02), xytext=(left_margin + train_w + val_w + test_w*0.5, y-0.005), arrowprops=dict(arrowstyle='->'))
ax1.text(left_margin + train_w + val_w + test_w*0.5, arrow_y - 0.04, 'Final metrics:\naccuracy, F1, ROC, NME, vertex error', ha='center', fontsize=9)

# Leakage annotation (clearer placement)
ax1.text(0.5, 0.93, 'No identity appears in more than one split', ha='center', fontsize=10, weight='bold', color='#222222')

# Panel label
ax1.text(0.01, 0.96, '(a) Face model evaluation protocol', fontsize=11, weight='bold')

# Panel 2: Crime forecasting temporal split (clean layout)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.axis('off')

# Blocks for train, val, hold-out with generous spacing
train_x = 0.04
train_w = 0.58
gap = 0.03
val_w = 0.25
hold_w = 0.10
h = 0.28
y = 0.55

ax2.add_patch(patches.Rectangle((train_x, y), train_w, h, facecolor='#a6cee3', edgecolor='k'))
ax2.add_patch(patches.Rectangle((train_x + train_w + gap, y), val_w, h, facecolor='#b2df8a', edgecolor='k'))
ax2.add_patch(patches.Rectangle((train_x + train_w + gap + val_w + gap, y), hold_w, h, facecolor='#fb9a99', edgecolor='k'))

ax2.text(train_x + train_w/2, y+h/2, 'Train period\n(months ...)', ha='center', va='center', fontsize=10, weight='bold')
ax2.text(train_x + train_w + gap + val_w/2, y+h/2, 'Validation period\n(hyperparam tuning)', ha='center', va='center', fontsize=10, weight='bold')
ax2.text(train_x + train_w + gap + val_w + gap + hold_w/2, y+h/2, 'Hold-out period\n(evaluation only)', ha='center', va='center', fontsize=10, weight='bold')

# Draw a long arrow below the timeline to emphasise no-future-data rule
arrow_y = y - 0.12
ax2.annotate('', xy=(train_x, arrow_y), xytext=(train_x + train_w + gap + val_w + gap + hold_w, arrow_y), arrowprops=dict(arrowstyle='<->', lw=1.2))
ax2.text(0.5, arrow_y - 0.06, 'No future data used for training', ha='center', fontsize=10, style='italic')

ax2.text(0.01, 0.96, '(b) Crime forecasting temporal split', fontsize=11, weight='bold')

# Save
plt.savefig(png_path, dpi=200)
plt.savefig(svg_path)
print('Saved:', png_path, svg_path)
plt.close(fig)
