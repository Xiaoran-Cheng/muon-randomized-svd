"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import argparse
import pickle
from itertools import repeat
import math
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass

#############################################
#               Muon optimizer              #
#############################################

@torch.compile
def quintic_ns_empirical(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T          #ax + bx^3 + cx^5
    return X



@torch.compile
def quintic_ns_theoretical(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (2.0, -1.5, 0.5)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T          #ax + bx^3 + cx^5
    return X


@torch.compile
def cubic_ns_theoretical(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b = (1.5, -0.5)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T          # ax + bx^3
    return X




coeffs_list = [
        (8.287212018145622, -23.595886519098816, 17.30038731253092),
        (4.1070591115422, -2.9478499167379084, 0.5448431082926599),
        (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
        (3.3184196573706033, -2.4884880243148757, 0.5100489401237204),
        (2.3006520199548173, -1.66890398457475, 0.4188073119525673),
        (1.8913014077874022, -1.2679958271945955, 0.37680408948525257),
        (1.8750014808698077, -1.2500016454327014, 0.37500016456381774),
        (1.8749999980503391, -1.2499999961006782, 0.3749999980503392),
        (1.875, -1.25, 0.375)
]


# safety factor for numerical stability (exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
               for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]


@torch.compile
def polar_express(G, steps= 3, eps = 1e-7):
    assert len(G.shape) == 2
    steps = int(steps)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


#############################################
#     Randomized Subspace Projection        #
#############################################

def randomized_project(M, rank=32, oversampling=2, power_iters=0):
    """
    Gaussian randomized subspace iteration.
    Returns Q (m x ell, orthonormal) and B = Q^T M (ell x n).

    Algorithm:
        1. Draw Gaussian sketch Omega (n x ell)
        2. Y = (MM^T)^h @ M @ Omega
        3. Q = orth(Y)
        4. B = Q^T @ M
    """
    m, n = M.shape
    ell = rank + oversampling

    # Sketch + power iteration in fp32 to avoid fp16 overflow
    # (sigma_max(M) cubed under power_iters>=1 routinely exceeds 65504 in fp16).
    M_f32 = M.float()
    Omega = torch.randn(n, ell, device=M.device, dtype=torch.float32)
    Y = M_f32 @ Omega
    for _ in range(power_iters):
        Y = M_f32 @ (M_f32.T @ Y)

    Q_f32, _ = torch.linalg.qr(Y)
    Q = Q_f32.to(M.dtype)
    B = (Q_f32.T @ M_f32).to(M.dtype)
    return Q, B

INEXACT_SOLVERS = (
    "polar_express",
    "cubic_ns_theoretical",
    "quintic_ns_theoretical",
    "quintic_ns_empirical",
)


SOLVER_FN = {
    "polar_express": polar_express,
    "cubic_ns_theoretical": cubic_ns_theoretical,
    "quintic_ns_theoretical": quintic_ns_theoretical,
    "quintic_ns_empirical": quintic_ns_empirical,
}


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False,
                 orth_method="quintic_ns_empirical", orth_steps=3,
                 randomized=False, rank=32, oversampling=2, power_iters=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        if orth_method not in INEXACT_SOLVERS:
            raise ValueError(f"Invalid orth_method: {orth_method}")
        if orth_steps < 0:
            raise ValueError(f"orth_steps must be non-negative, got {orth_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            orth_method=orth_method,
            orth_steps=int(orth_steps),
            randomized=bool(randomized),
            rank=int(rank),
            oversampling=int(oversampling),
            power_iters=int(power_iters),
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            orth_method = group["orth_method"]
            orth_steps = int(group["orth_steps"])
            randomized = group["randomized"]
            rank = int(group["rank"])
            oversampling = int(group["oversampling"])
            power_iters = int(group["power_iters"])
            solver_fn = SOLVER_FN[orth_method]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                g2d = g.reshape(len(g), -1)

                if randomized:
                    # 1. Project: Q (m x ell), B (ell x n)
                    Q, B = randomized_project(g2d, rank=rank,
                                              oversampling=oversampling,
                                              power_iters=power_iters)
                    # 2. Solver on compressed B
                    Z = solver_fn(B, steps=orth_steps, eps=1e-7)
                    # 3. Lift back: T = Q @ Z
                    update_2d = Q.bfloat16() @ Z
                else:
                    update_2d = solver_fn(g2d, steps=orth_steps, eps=1e-7)

                update = update_2d.view(g.shape)
                p.data.add_(update, alpha=-lr)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "tta_test_acc", "time_seconds"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    was_training = model.training
    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        logits = torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])
    if was_training:
        model.train()
    return logits

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate_with_loss(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    val_loss = F.cross_entropy(logits, loader.labels, reduction="mean").item()
    val_acc = (logits.argmax(1) == loader.labels).float().mean().item()
    return val_loss, val_acc


FOLDER_ARGS = [
    "optimizer_mode",
    "batch_size",
    "muon_lr",
    "muon_momentum",
    "muon_nesterov",
    "inexact_solver",
    "orth_steps",
    "randomized",
    "rank",
    "oversampling",
    "power_iters",
]

ARG_ABBREVIATIONS = {
    "optimizer_mode": "om",
    "inexact_solver": "is",
    "orth_steps": "os",
    "batch_size": "bs",
    "muon_lr": "mlr",
    "muon_momentum": "mmm",
    "muon_nesterov": "mn",
    "randomized": "rz",
    "rank": "rk",
    "oversampling": "ov",
    "power_iters": "pi",
}


def _format_arg_value(value):
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def build_output_dir_from_args(args):
    arg_items = vars(args)
    parts = []
    for key in FOLDER_ARGS:
        abbr = ARG_ABBREVIATIONS.get(key, key)
        parts.append(f"{abbr}{_format_arg_value(arg_items[key])}")
    folder_name = "_".join(parts)
    variant_subdir = getattr(args, "wandb_project", None) or "default"
    output_dir = os.path.join("logs", variant_subdir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_console_path(path):
    abs_path = os.path.abspath(path)
    encoding = sys.stdout.encoding or "utf-8"
    return abs_path.encode(encoding, errors="backslashreplace").decode(encoding)

############################################
#                Training                  #
############################################

def main(run, model, optimizer_mode, orth_method, orth_steps, batch_size, epochs, val_every_steps,
         sgd_momentum, sgd_nesterov,
         muon_lr, muon_momentum, muon_nesterov,
         filter_sgd_lr, filter_sgd_weight_decay,
         adamw_beta1, adamw_beta2, adamw_eps, filter_adamw_lr, filter_adamw_weight_decay,
         randomized, rank, oversampling, power_iters,
         use_wandb=False):
    optimizer_mode = str(optimizer_mode)
    batch_size = int(batch_size)
    epochs = int(epochs)
    val_every_steps = int(val_every_steps)
    sgd_momentum = float(sgd_momentum)
    sgd_nesterov = bool(sgd_nesterov)
    muon_lr = float(muon_lr)
    muon_momentum = float(muon_momentum)
    muon_nesterov = bool(muon_nesterov)
    filter_sgd_lr = float(filter_sgd_lr)
    filter_sgd_weight_decay = float(filter_sgd_weight_decay)
    adamw_beta1 = float(adamw_beta1)
    adamw_beta2 = float(adamw_beta2)
    adamw_eps = float(adamw_eps)
    filter_adamw_lr = float(filter_adamw_lr)
    filter_adamw_weight_decay = float(filter_adamw_weight_decay)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if val_every_steps <= 0:
        raise ValueError(f"val_every_steps must be positive, got {val_every_steps}")
    if optimizer_mode not in ("muon", "sgd", "adamw"):
        raise ValueError(f"optimizer_mode must be one of muon/sgd/adamw, got {optimizer_mode}")
    if sgd_momentum < 0:
        raise ValueError(f"sgd_momentum must be non-negative, got {sgd_momentum}")
    if optimizer_mode in ("muon", "sgd") and sgd_nesterov and sgd_momentum <= 0:
        raise ValueError("sgd_nesterov requires sgd_momentum > 0")
    if optimizer_mode == "muon":
        if muon_lr < 0:
            raise ValueError(f"muon_lr must be non-negative, got {muon_lr}")
        if muon_momentum < 0:
            raise ValueError(f"muon_momentum must be non-negative, got {muon_momentum}")
    if optimizer_mode == "sgd":
        if filter_sgd_lr < 0:
            raise ValueError(f"filter_sgd_lr must be non-negative, got {filter_sgd_lr}")
        if filter_sgd_weight_decay < 0:
            raise ValueError(
                f"filter_sgd_weight_decay must be non-negative, got {filter_sgd_weight_decay}"
            )
    if optimizer_mode == "adamw":
        if not 0 <= adamw_beta1 < 1:
            raise ValueError(f"adamw_beta1 must be in [0, 1), got {adamw_beta1}")
        if not 0 <= adamw_beta2 < 1:
            raise ValueError(f"adamw_beta2 must be in [0, 1), got {adamw_beta2}")
        if adamw_eps < 0:
            raise ValueError(f"adamw_eps must be non-negative, got {adamw_eps}")
        if filter_adamw_lr < 0:
            raise ValueError(f"filter_adamw_lr must be non-negative, got {filter_adamw_lr}")
        if filter_adamw_weight_decay < 0:
            raise ValueError(
                "filter_adamw_weight_decay must be non-negative, "
                f"got {filter_adamw_weight_decay}"
            )

    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    full_train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))

    # Split 5000 samples from training set as a held-out validation set
    n_val = 5000
    n_total = len(full_train_loader.images)
    perm = torch.randperm(n_total, device=full_train_loader.images.device)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    val_loader = CifarLoader("cifar10", train=True, batch_size=2000)
    val_loader.images = full_train_loader.images[val_indices]
    val_loader.labels = full_train_loader.labels[val_indices]
    val_loader.aug = {}
    val_loader.drop_last = False
    val_loader.shuffle = False

    train_loader = full_train_loader
    train_loader.images = full_train_loader.images[train_indices]
    train_loader.labels = full_train_loader.labels[train_indices]

    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(epochs * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    non_filter_param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases,         lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    if optimizer_mode == "muon":
        optimizer1 = torch.optim.SGD(
            non_filter_param_configs,
            momentum=sgd_momentum,
            nesterov=sgd_nesterov,
            fused=True,
        )
        optimizer2 = Muon(
            filter_params,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            orth_method=orth_method,
            orth_steps=orth_steps,
            randomized=randomized,
            rank=rank,
            oversampling=oversampling,
            power_iters=power_iters,
        )
        optimizers = [optimizer1, optimizer2]
        whiten_bias_lr_groups = optimizer1.param_groups[:1]
        decay_lr_groups = optimizer1.param_groups[1:] + optimizer2.param_groups
    elif optimizer_mode == "sgd":
        optimizer = torch.optim.SGD(
            non_filter_param_configs + [
                dict(
                    params=filter_params,
                    lr=filter_sgd_lr,
                    weight_decay=filter_sgd_weight_decay,
                )
            ],
            momentum=sgd_momentum,
            nesterov=sgd_nesterov,
            fused=True,
        )
        optimizers = [optimizer]
        whiten_bias_lr_groups = optimizer.param_groups[:1]
        decay_lr_groups = optimizer.param_groups[1:]
    else:
        optimizer = torch.optim.AdamW(
            [
                dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd),
                dict(params=norm_biases,         lr=bias_lr, weight_decay=wd),
                dict(params=[model.head.weight], lr=head_lr, weight_decay=wd),
                dict(
                    params=filter_params,
                    lr=filter_adamw_lr,
                    weight_decay=filter_adamw_weight_decay,
                ),
            ],
            betas=(adamw_beta1, adamw_beta2),
            eps=adamw_eps,
            fused=True,
        )
        optimizers = [optimizer]
        whiten_bias_lr_groups = optimizer.param_groups[:1]
        decay_lr_groups = optimizer.param_groups[1:]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0
    eval_steps = []
    val_losses = []
    val_accs = []
    train_losses = []
    train_accs = []

    # Running accumulators for training stats between eval points
    window_loss_sum = 0.0
    window_correct = 0
    window_count = 0
    train_loss = float("nan")
    train_acc = float("nan")

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = F.cross_entropy(outputs.float(), labels, label_smoothing=0.2, reduction="sum")
            loss.backward()
            batch_loss_sum = loss.item()
            if math.isfinite(batch_loss_sum):
                window_loss_sum += batch_loss_sum
                window_correct += (outputs.detach().argmax(1) == labels).sum().item()
                window_count += len(labels)
            for group in whiten_bias_lr_groups:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in decay_lr_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            should_eval = (step % val_every_steps == 0) or (step >= total_train_steps)
            if should_eval:
                val_loss_step, val_acc_step = evaluate_with_loss(model, val_loader, tta_level=0)
                if window_count > 0:
                    train_loss = window_loss_sum / window_count
                    train_acc = window_correct / window_count
                eval_steps.append(step)
                val_losses.append(val_loss_step)
                val_accs.append(val_acc_step)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                if use_wandb:
                    wandb.log({
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss_step,
                        "val_acc": val_acc_step,
                        "lr": decay_lr_groups[0]["lr"],
                        "step": step,
                    })
                window_loss_sum = 0.0
                window_correct = 0
                window_count = 0
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        if eval_steps and eval_steps[-1] == step:
            val_acc = val_accs[-1]
        else:
            val_acc = evaluate(model, val_loader, tta_level=0)
        if use_wandb:
            wandb.log({"epoch": epoch})
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, val_loader, tta_level=2)
    tta_test_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    if use_wandb:
        wandb.summary["final_val_acc"] = tta_val_acc
        wandb.summary["final_test_acc"] = tta_test_acc
        wandb.summary["time_seconds"] = time_seconds

    return {
        "eval_steps": eval_steps,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "final_val_acc": tta_val_acc,
        "final_test_acc": tta_test_acc,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer-mode", "--optimizer_mode",
        type=str,
        default="muon",
        choices=("muon", "sgd", "adamw"),
        help="Optimizer mode: split SGD+Muon, single SGD, or single AdamW.",
    )
    parser.add_argument(
        "--inexact-solver", "--inexact_solver",
        type=str,
        default="quintic_ns_empirical",
        choices=INEXACT_SOLVERS,
        help="Inexact solver used in Muon.step.",
    )
    parser.add_argument(
        "--orth-steps", "--orth_steps",
        type=int,
        default=3,
        help="Number of iterations passed to the selected inexact solver.",
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        type=int,
        default=2000,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--sgd-momentum", "--sgd_momentum",
        type=float,
        default=0.85,
        help="Momentum used by SGD parameter groups.",
    )
    parser.add_argument(
        "--sgd-nesterov", "--sgd_nesterov",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable Nesterov momentum for SGD parameter groups (True/False).",
    )
    parser.add_argument(
        "--muon-lr", "--muon_lr",
        type=float,
        default=0.24,
        help="Learning rate for Muon optimizer.",
    )
    parser.add_argument(
        "--muon-momentum", "--muon_momentum",
        type=float,
        default=0.6,
        help="Momentum for Muon optimizer.",
    )
    parser.add_argument(
        "--muon-nesterov", "--muon_nesterov",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable Nesterov momentum for Muon (True/False).",
    )
    parser.add_argument(
        "--filter-sgd-lr", "--filter_sgd_lr",
        type=float,
        default=0.24,
        help="Learning rate for 4D filter weights in SGD mode.",
    )
    parser.add_argument(
        "--filter-sgd-weight-decay", "--filter_sgd_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for 4D filter weights in SGD mode.",
    )
    parser.add_argument(
        "--adamw-beta1", "--adamw_beta1",
        type=float,
        default=0.9,
        help="Beta1 parameter for AdamW mode.",
    )
    parser.add_argument(
        "--adamw-beta2", "--adamw_beta2",
        type=float,
        default=0.999,
        help="Beta2 parameter for AdamW mode.",
    )
    parser.add_argument(
        "--adamw-eps", "--adamw_eps",
        type=float,
        default=1e-8,
        help="Epsilon parameter for AdamW mode.",
    )
    parser.add_argument(
        "--filter-adamw-lr", "--filter_adamw_lr",
        type=float,
        default=0.24,
        help="Learning rate for 4D filter weights in AdamW mode.",
    )
    parser.add_argument(
        "--filter-adamw-weight-decay", "--filter_adamw_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for 4D filter weights in AdamW mode.",
    )
    parser.add_argument(
        "--val-every-steps", "--val_every_steps",
        type=int,
        default=100,
        help="Evaluate validation loss/accuracy every N training steps.",
    )
    parser.add_argument(
        "--num-trials", "--num_trials",
        type=int,
        default=200,
        help="Number of measured training trials after warmup.",
    )
    # Randomized solver parameters
    parser.add_argument(
        "--randomized",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Enable randomized subspace projection before the solver (True/False).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Target rank k for randomized projection.",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=2,
        help="Oversampling parameter p for randomized projection (p >= 2).",
    )
    parser.add_argument(
        "--power-iters", "--power_iters",
        type=int,
        default=0,
        help="Number of power iterations h for randomized projection.",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Enable Weights & Biases logging (True/False).",
    )
    parser.add_argument(
        "--wandb-project", "--wandb_project",
        dest="wandb_project",
        type=str,
        default="cifar10-muon",
        help="W&B project name (ignored inside a sweep agent; sweep's project wins).",
    )
    parser.add_argument(
        "--wandb-group", "--wandb_group",
        dest="wandb_group",
        type=str,
        default=None,
        help="W&B group name; defaults to output_dir basename when unset.",
    )
    args = parser.parse_args()
    if args.num_trials <= 0:
        raise ValueError(f"--num-trials must be positive, got {args.num_trials}")
    if args.val_every_steps <= 0:
        raise ValueError(f"--val-every-steps must be positive, got {args.val_every_steps}")
    if args.optimizer_mode in ("muon", "sgd") and args.sgd_momentum < 0:
        raise ValueError(f"--sgd-momentum must be non-negative, got {args.sgd_momentum}")
    if args.optimizer_mode in ("muon", "sgd") and args.sgd_nesterov and args.sgd_momentum <= 0:
        raise ValueError("--sgd-nesterov requires --sgd-momentum > 0")
    if args.optimizer_mode == "muon":
        if args.muon_lr < 0:
            raise ValueError(f"--muon-lr must be non-negative, got {args.muon_lr}")
        if args.muon_momentum < 0:
            raise ValueError(f"--muon-momentum must be non-negative, got {args.muon_momentum}")
    if args.optimizer_mode == "sgd":
        if args.filter_sgd_lr < 0:
            raise ValueError(f"--filter-sgd-lr must be non-negative, got {args.filter_sgd_lr}")
        if args.filter_sgd_weight_decay < 0:
            raise ValueError(
                "--filter-sgd-weight-decay must be non-negative, "
                f"got {args.filter_sgd_weight_decay}"
            )
    if args.optimizer_mode == "adamw":
        if not 0 <= args.adamw_beta1 < 1:
            raise ValueError(f"--adamw-beta1 must be in [0, 1), got {args.adamw_beta1}")
        if not 0 <= args.adamw_beta2 < 1:
            raise ValueError(f"--adamw-beta2 must be in [0, 1), got {args.adamw_beta2}")
        if args.adamw_eps < 0:
            raise ValueError(f"--adamw-eps must be non-negative, got {args.adamw_eps}")
        if args.filter_adamw_lr < 0:
            raise ValueError(f"--filter-adamw-lr must be non-negative, got {args.filter_adamw_lr}")
        if args.filter_adamw_weight_decay < 0:
            raise ValueError(
                "--filter-adamw-weight-decay must be non-negative, "
                f"got {args.filter_adamw_weight_decay}"
            )

    use_wandb = bool(args.wandb)
    if use_wandb and not _wandb_available:
        raise RuntimeError("--wandb requires the `wandb` package; install it with `pip install wandb`")

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    output_dir = build_output_dir_from_args(args)
    print(f"Saving outputs to: {safe_console_path(output_dir)}")

    print_columns(logging_columns_list, is_head=True)
    main(
        "warmup",
        model,
        args.optimizer_mode,
        args.inexact_solver,
        args.orth_steps,
        args.batch_size,
        args.epochs,
        args.val_every_steps,
        args.sgd_momentum,
        args.sgd_nesterov,
        args.muon_lr,
        args.muon_momentum,
        args.muon_nesterov,
        args.filter_sgd_lr,
        args.filter_sgd_weight_decay,
        args.adamw_beta1,
        args.adamw_beta2,
        args.adamw_eps,
        args.filter_adamw_lr,
        args.filter_adamw_weight_decay,
        args.randomized,
        args.rank,
        args.oversampling,
        args.power_iters,
        use_wandb=False,
    )
    trial_outputs = []
    wandb_parent_dir = os.path.dirname(output_dir)
    if use_wandb:
        os.makedirs(wandb_parent_dir, exist_ok=True)
    for run in range(args.num_trials):
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                group=(args.wandb_group or os.path.basename(output_dir))[:128],
                name=f"trial_{run:03d}",
                reinit=True,
                dir=wandb_parent_dir,
            )
        trial_result = main(
            run,
            model,
            args.optimizer_mode,
            args.inexact_solver,
            args.orth_steps,
            args.batch_size,
            args.epochs,
            args.val_every_steps,
            args.sgd_momentum,
            args.sgd_nesterov,
            args.muon_lr,
            args.muon_momentum,
            args.muon_nesterov,
            args.filter_sgd_lr,
            args.filter_sgd_weight_decay,
            args.adamw_beta1,
            args.adamw_beta2,
            args.adamw_eps,
            args.filter_adamw_lr,
            args.filter_adamw_weight_decay,
            args.randomized,
            args.rank,
            args.oversampling,
            args.power_iters,
            use_wandb=use_wandb,
        )
        if use_wandb:
            wandb.finish()
        trial_outputs.append(trial_result)
        model_path = os.path.join(output_dir, f"trial_{run:03d}_model.pt")
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "trial": run,
                "args": vars(args),
            },
            model_path,
        )

    eval_steps = trial_outputs[0]["eval_steps"]
    num_eval_points = len(eval_steps)
    validation_columns = []
    validation_values = torch.empty((num_eval_points, 4 * len(trial_outputs)), dtype=torch.float32)
    for trial_idx, trial_output in enumerate(trial_outputs):
        if trial_output["eval_steps"] != eval_steps:
            raise ValueError("Validation steps differ across trials; cannot build aligned validation_metrics.pkl")
        validation_values[:, 4 * trial_idx]     = torch.tensor(trial_output["train_losses"], dtype=torch.float32)
        validation_values[:, 4 * trial_idx + 1] = torch.tensor(trial_output["train_accs"],   dtype=torch.float32)
        validation_values[:, 4 * trial_idx + 2] = torch.tensor(trial_output["val_losses"],   dtype=torch.float32)
        validation_values[:, 4 * trial_idx + 3] = torch.tensor(trial_output["val_accs"],    dtype=torch.float32)
        validation_columns.extend([
            f"trial_{trial_idx:03d}_train_loss",
            f"trial_{trial_idx:03d}_train_acc",
            f"trial_{trial_idx:03d}_val_loss",
            f"trial_{trial_idx:03d}_val_acc",
        ])

    validation_metrics_payload = {
        "eval_steps": eval_steps,
        "columns": validation_columns,
        "values": validation_values,
    }
    validation_metrics_path = os.path.join(output_dir, "validation_metrics.pkl")
    with open(validation_metrics_path, "wb") as f:
        pickle.dump(validation_metrics_payload, f)

    val_accs_final = torch.tensor([out["final_val_acc"] for out in trial_outputs], dtype=torch.float32)
    test_accs_final = torch.tensor([out["final_test_acc"] for out in trial_outputs], dtype=torch.float32)
    accuracy_payload = {
        "rows": [f"trial_{i:03d}" for i in range(len(trial_outputs))],
        "columns": ["val_acc_tta2", "test_acc_tta2"],
        "values": torch.stack([val_accs_final, test_accs_final], dim=1),
    }
    accuracy_path = os.path.join(output_dir, "accuracy.pkl")
    with open(accuracy_path, "wb") as f:
        pickle.dump(accuracy_payload, f)

    print("Val  Mean: %.4f    Std: %.4f" % (val_accs_final.mean(), val_accs_final.std()))
    print("Test Mean: %.4f    Std: %.4f" % (test_accs_final.mean(), test_accs_final.std()))

    log_path = os.path.join(output_dir, "log.pt")
    torch.save(dict(code=code, val_accs=val_accs_final, test_accs=test_accs_final, args=vars(args)), log_path)
    print(safe_console_path(log_path))