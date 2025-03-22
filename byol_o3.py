import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import copy
from argparse import ArgumentParser

parser = argparse.ArgumentParser(description="Parse the number of epochs.")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, hidden_dim=4096, m=0.996):
        super(BYOL, self).__init__()
        self.m = m
        self.online_encoder = base_encoder()
        if hasattr(self.online_encoder, 'fc'):
            in_features = self.online_encoder.fc.in_features
            self.online_encoder.fc = nn.Identity()
        else:
            raise ValueError("No fc layer found!")
        self.online_projector = MLP(in_features, hidden_dim, projection_dim)
        self.online_predictor = MLP(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._set_requires_grad(self.target_encoder, False)
        self._set_requires_grad(self.target_projector, False)
    def _set_requires_grad(self, module, flag=False):
        for p in module.parameters():
            p.requires_grad = flag
    @torch.no_grad()
    def update_target(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.m + online_params.data * (1.0 - self.m)
        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = target_params.data * self.m + online_params.data * (1.0 - self.m)
    def forward(self, view1, view2):
        online_repr1 = self.online_encoder(view1)
        online_proj1 = self.online_projector(online_repr1)
        online_pred1 = self.online_predictor(online_proj1)
        online_repr2 = self.online_encoder(view2)
        online_proj2 = self.online_projector(online_repr2)
        online_pred2 = self.online_predictor(online_proj2)
        with torch.no_grad():
            target_repr1 = self.target_encoder(view1)
            target_proj1 = self.target_projector(target_repr1)
            target_repr2 = self.target_encoder(view2)
            target_proj2 = self.target_projector(target_repr2)
        return online_pred1, online_pred2, target_proj1.detach(), target_proj2.detach()

def byol_loss(pred, target):
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 2 - 2 * (pred_norm * target_norm).sum(dim=-1).mean()

class BYOLTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

def main(args):

    num_epochs= args.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    byol_transform = BYOLTransform()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=byol_transform)
    def collate_fn(batch):
        view1_list, view2_list = [], []
        for (v1, v2), _ in batch:
            view1_list.append(v1)
            view2_list.append(v2)
        view1_batch = torch.stack(view1_list, 0)
        view2_batch = torch.stack(view2_list, 0)
        return view1_batch, view2_batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_fn)
    model = BYOL(base_encoder=lambda: resnet18(num_classes=10), projection_dim=256, hidden_dim=4096, m=0.996)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
     
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (view1, view2) in enumerate(train_loader):
            view1 = view1.to(device)
            view2 = view2.to(device)
            optimizer.zero_grad()
            online_pred1, online_pred2, target_proj1, target_proj2 = model(view1, view2)
            loss1 = byol_loss(online_pred1, target_proj2)
            loss2 = byol_loss(online_pred2, target_proj1)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.update_target()
            epoch_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:3d} Batch {batch_idx:4d}: Loss {loss.item():.4f}")
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d} completed: Avg Loss {avg_loss:.4f}")
    print("Training complete.")
    print("saving model....")
    torch.save(model.state_dict(), 'byol_o3.pth')

if __name__ == "__main__":
    main(args)
