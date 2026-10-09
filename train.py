"""Train PA_CAE."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import XiamenBus
from pa_cae import PA_CAE
from losses import DmdMreLoss

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', default='runs/pa_cae')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--beta', type=float, default=0.)
    p.add_argument('--seed', type=int, default=3035)
    p.add_argument('--device', default='cpu')
    p.add_argument('--imagenet-pretrained', action='store_true')
    a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    model=PA_CAE(pretrained=a.imagenet_pretrained).to(a.device)
    loader=DataLoader(XiamenBus(a.data_root, 'train', augment=True), batch_size=a.batch_size, shuffle=True)
    loss_fn=DmdMreLoss(a.alpha,a.beta)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.995)
    for epoch in range(a.epochs):
        model.train(); total=0.
        for x,y,_ in loader:
            optimizer.zero_grad()
            pred=model(x.to(a.device))
            loss,_,_=loss_fn(pred,y.to(a.device)*100.)
            loss.backward(); optimizer.step(); total+=loss.item()
        scheduler.step()
        torch.save(model.state_dict(),out/'pa_cae_latest.pth')
        print(f'Epoch {epoch+1}: loss={total/len(loader):.6f}')
if __name__=='__main__': main()
