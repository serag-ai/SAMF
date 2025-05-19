from medmnist import OrganAMNIST, OrganCMNIST, OrganSMNIST
from torch.utils.data import ConcatDataset


def GetMergedMedMNISTDataset(transform, target_transform, roots):

    val_ora_dt = OrganAMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="val",
        root=roots["OrganAMNIST"],
        as_rgb=False,
        size=224,
    )
    train_ora_dt = OrganAMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="train",
        root=roots["OrganAMNIST"],
        as_rgb=False,
        size=224,
    )
    test_ora_dt = OrganAMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="test",
        root=roots["OrganAMNIST"],
        as_rgb=False,
        size=224,
    )

    val_ors_dt = OrganSMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="val",
        root=roots["OrganSMNIST"],
        as_rgb=False,
        size=224,
    )
    train_ors_dt = OrganSMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="train",
        root=roots["OrganSMNIST"],
        as_rgb=False,
        size=224,
    )
    test_ors_dt = OrganSMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="test",
        root=roots["OrganSMNIST"],
        as_rgb=False,
        size=224,
    )

    val_orc_dt = OrganCMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="val",
        root=roots["OrganCMNIST"],
        as_rgb=False,
        size=224,
    )
    test_orc_dt = OrganCMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="test",
        root=roots["OrganCMNIST"],
        as_rgb=False,
        size=224,
    )
    train_orc_dt = OrganCMNIST(
        transform=transform,
        target_transform=target_transform,
        download=True,
        split="train",
        root=roots["OrganCMNIST"],
        as_rgb=False,
        size=224,
    )
    return ConcatDataset(
        [
            val_ora_dt,
            train_ora_dt,
            test_ora_dt,
            val_orc_dt,
            train_orc_dt,
            test_orc_dt,
            train_ors_dt,
            val_ors_dt,
            test_ors_dt,
        ]
    )
