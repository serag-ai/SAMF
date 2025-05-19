import copy

import torch

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import os
import torchvision
import argparse
from dataset import GetMergedMedMNISTDataset


def target_transform(t):
    return 0


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DINO model with specified hyperparameters."
    )

    # Model hyperparameters
    parser.add_argument("--output_dim", type=int, default=2048, help="Model dimension.")

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning Rate.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument(
        "--evaluation_epochs", type=int, default=50, help="Evluation epochs."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )

    args = parser.parse_args()

    backbone = torch.hub.load(
        "facebookresearch/dino:main", "dino_vitb16", pretrained=False
    )

    input_dim = backbone.embed_dim
    print(f"INPUT DIM IS: {input_dim}")
    model = DINO(backbone, input_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=3),
            DINOTransform(normalize=None),
        ]
    )

    dataset = GetMergedMedMNISTDataset(
        transform=transform,
        target_transform=target_transform,
        roots={
            "OrganAMNIST": "./datasets/medmnist/organAmnist/",
            "OrganSMNIST": "./datasets/medmnist/organSmnist/",
            "OrganCMNIST": "./datasets/medmnist/organCmnist/",
        },
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )

    criterion = DINOLoss(
        output_dim=args.output_dim,
        warmup_teacher_temp_epochs=5,
    )
    # move loss to correct device because it also contains parameters
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting Training")
    for epoch in range(args.epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, args.epochs, 0.996, 1)
        for batch in dataloader:
            views = batch[0]
            update_momentum(
                model.student_backbone, model.teacher_backbone, m=momentum_val
            )
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        if ((epoch) % args.evaluation_epochs) == 0:
            torch.save(
                model,
                os.path.join(args.output_dir, f"dino_all__{epoch}.pth"),
            )
            torch.save(
                model.student_backbone.state_dict(),
                os.path.join(args.output_dir, f"dino_backbone_{epoch}.pth"),
            )

        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "dino_all.pth"))
