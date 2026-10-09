"""Export per-image predictions; reject incompatible checkpoints strictly."""
import argparse, csv, json, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import XiamenBus
from pa_cae import PA_CAE

def load_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = state.get('net', state.get('state_dict', state))
    clean = {}
    for k, v in state.items():
        if k.startswith('module.'): k = k[7:]
        if k.startswith('CCN.'): k = k[4:]
        clean[k] = v
    model.load_state_dict(clean, strict=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--model', choices=['pa_cae', 'resnet101'], default='pa_cae')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='predictions.csv')
    a = p.parse_args()
    if a.model == 'resnet101':
        from resnet101 import Res101_SFCN
        model = Res101_SFCN(pretrained=False)
    else:
        model = PA_CAE(pretrained=False)
    load_weights(model, a.weights)
    model.to(a.device).eval()
    rows = []
    with torch.no_grad():
        for x, y, names in DataLoader(XiamenBus(a.data_root), batch_size=1):
            pred = model(x.to(a.device)).sum().item() / 100.
            gt = y.sum().item()
            rows.append(dict(image=names[0], ground_truth=gt, prediction=pred, absolute_error=abs(pred-gt)))
    if not rows: raise ValueError('Empty test split')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    errors = [r['absolute_error'] for r in rows]
    metrics = dict(n=len(rows), MAE=sum(errors)/len(rows), RMSE=math.sqrt(sum(e*e for e in errors)/len(rows)),
                   model=a.model, thresholds={str(t):sum(e<t for e in errors) for t in (1,3,5)})
    out.with_suffix('.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
if __name__ == '__main__': main()
