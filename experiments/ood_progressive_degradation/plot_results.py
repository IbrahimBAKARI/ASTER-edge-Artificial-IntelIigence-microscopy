from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
df = pd.read_csv(root/'results/ood_scores.csv')
out = root/'results'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':9,
                     'axes.labelsize':8, 'legend.fontsize':7, 'xtick.labelsize':7, 'ytick.labelsize':7})
fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), layout='constrained')
for column, family in enumerate(['blur','color']):
    frame = df[df.family.eq(family)]
    for diagnosis, label, color, marker, style in [
        ('Stem cell donor','Donors (n = 20)','#0072B2','o','-'),
        ('AML','AML (n = 10)','#D55E00','s','--')]:
        sub = frame[frame.diagnosis.eq(diagnosis)]
        grouped = sub.groupby('level').ood_score
        x = sorted(sub.level.unique())
        axes[0,column].plot(x,grouped.median(),color=color,marker=marker,linestyle=style,markersize=3,label=label)
        axes[0,column].fill_between(x,grouped.quantile(.25),grouped.quantile(.75),color=color,alpha=.15)
        axes[1,column].plot(x,sub.groupby('level').rejected.mean()*100,color=color,marker=marker,linestyle=style,markersize=3,label=label)
    axes[0,column].axhline(576.1117745962939,color='black',linestyle=':',linewidth=1,label='Frozen threshold')
    axes[0,column].set_title(('(a) Gaussian blur','(b) Color perturbation')[column],loc='left')
    axes[1,column].set_title(('(c) Gaussian blur','(d) Color perturbation')[column],loc='left')
    axes[0,column].set_ylabel('Median bag OOD score (IQR)')
    axes[1,column].set_ylabel('Rejected patients (%)')
    axes[1,column].set_ylim(-2,102)
    for row in range(2):
        ax = axes[row,column]
        ax.set_xlabel('Blur radius (pixels)' if family=='blur' else 'Color perturbation level')
        ax.grid(axis='y',alpha=.2)
        ax.legend(loc='upper left' if column==0 else 'best')
fig.savefig(out/'ood_progressive_degradation.png',dpi=300)
fig.savefig(out/'ood_progressive_degradation.pdf')
