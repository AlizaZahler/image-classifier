import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from .data import CLASSES, load_rgb
from .metrics import classification_metrics

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def plots(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for stage in sorted({r['stage'] for r in history}):
        rows = [r for r in history if r['stage'] == stage]
        axes[0].plot([r['epoch'] for r in rows], [r['train']['loss'] for r in rows], label=stage)
        val = [r for r in rows if r.get('validation')]
        if val:
            axes[1].plot([r['epoch'] for r in val], [r['validation']['macro_f1'] for r in val], label=stage)
    for ax, title in zip(axes, ('Training loss', 'Validation macro-F1')):
        ax.set(xlabel='Epoch', title=title)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(output) / 'training_curves.png', dpi=140)
    plt.close(fig)

def evaluation_report(root, rows, metrics, predictions, output, checkpoint):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(CLASSES)
    metrics['checkpoint'] = str(checkpoint)
    metrics['interpretation'] = ('Performance on represented source styles. Productions and source styles overlap '
                                 'across outer splits; this does not measure generalization to unseen productions. '
                                 'Softmax outputs are uncalibrated model scores.')
    metrics['group_metrics_note'] = 'Macro-F1 uses all three labels, including absent labels; see per-class counts.'
    for field in ('source', 'subtype', 'source_work'):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row.get(field) or '(unspecified)'].append(i)
        metrics['by_' + field] = {key: classification_metrics(
            [predictions['targets'][i] for i in ids], [predictions['predictions'][i] for i in ids])
            for key, ids in groups.items()}
    write_json(output / 'metrics.json', metrics)
    with (output / 'predictions.csv').open('w', newline='') as f:
        fields = ['image_id', 'file', 'label', 'predicted', 'source', 'subtype', 'source_work', 'score_2d', 'score_3d', 'score_live']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            record = {k: row.get(k, '') for k in fields[:7]}
            record['predicted'] = names[predictions['predictions'][i]]
            record.update(zip(fields[7:], predictions['scores'][i]))
            writer.writerow(record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = metrics['confusion_matrix']
    ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
           xlabel='Predicted', ylabel='Actual', title='Test confusion matrix')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    thumbs = output / 'mistakes'
    thumbs.mkdir(exist_ok=True)
    cards = []
    for i, row in enumerate(rows):
        if predictions['targets'][i] == predictions['predictions'][i]:
            continue
        im = load_rgb(Path(root) / row['file'])
        im.thumbnail((240, 180))
        im.save(thumbs / f'{i}.jpg')
        caption = html.escape(f"{row['image_id']}: {row['label']} → {names[predictions['predictions'][i]]}")
        cards.append(f'<figure><img src="mistakes/{i}.jpg"><figcaption>{caption}</figcaption></figure>')
    (output / 'mistakes.html').write_text('<!doctype html><meta charset="utf-8"><title>Classification mistakes</title>'
        '<style>body{font:16px sans-serif}main{display:flex;flex-wrap:wrap}figure{width:250px}</style>'
        '<h1>Test mistakes</h1><p>Uncalibrated scores; represented source styles only.</p><main>'
        + ''.join(cards) + '</main>')
