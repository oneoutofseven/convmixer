import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from convmixer import ConvMixer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

#######################################
# 1. 数据预处理（BreaKHis 建议 Resize 到 256 再 crop 224）
#######################################

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

#######################################
# 2. 直接用 ImageFolder 读取 train / test
# 路径必须是：
#   BreaKHis_400X/train/benign/
#   BreaKHis_400X/train/malignant/
#   BreaKHis_400X/test/benign/
#   BreaKHis_400X/test/malignant/
#######################################

train_root = "./BreaKHis_400X/train"
test_root  = "./BreaKHis_400X/test"

train_full = datasets.ImageFolder(train_root, transform=train_tf)
test_set   = datasets.ImageFolder(test_root,  transform=test_tf)

num_classes = len(train_full.classes)
print("Detected classes:", train_full.classes, "→ num_classes =", num_classes)

#######################################
# 3. 从 train_full 划分出 train / val（8:2）
#######################################

train_size = int(0.8 * len(train_full))
val_size   = len(train_full) - train_size

train_set, val_set = random_split(train_full, [train_size, val_size])

# val 用 test transform（不做增强）
val_set.dataset.transform = test_tf

#######################################
# 4. Dataloader
#######################################

train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)

#######################################
# 5. ConvMixer 模型
#######################################

model = ConvMixer(
    dim=256,        # 显存不够就改成 128
    depth=8,        # 越多越强，但也越耗显存
    kernel_size=7,
    patch_size=7,
    n_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

#######################################
# 6. 验证函数
#######################################

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

#######################################
# 7. 训练循环
#######################################

best_val_acc = 0
best_path = "best_convmixer_breakhis.pth"

for epoch in range(30):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    scheduler.step()

    train_loss = running_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(val_loader)

    print(f"Epoch {epoch+1:02d}/30 | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  >> Best model saved (val_acc={val_acc:.4f})")

#######################################
# 8. 测试集最终结果
#######################################

model.load_state_dict(torch.load(best_path))
test_loss, test_acc = evaluate(test_loader)
print(f"\nTest result: loss={test_loss:.4f}, acc={test_acc:.4f}")