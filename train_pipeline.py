import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image

# ==========================================
# [글로벌 규격 일치 - main.py와 100% 동기화]
# ==========================================
TARGET_MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TARGET_OBJECTS = ["수정테이프", "페트병", "종이컵", "음료수캔", "골판지상자", "일반비닐", "가위", "기타물품"]
MODEL_NAME = "best_ecovision_multitask.pth"
DATASET_DIR = "dataset"

# 데이터셋 폴더 초기화 방어 로직
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)
    for cat in TARGET_MATERIALS:
        os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)
    print(f"⚠ '{DATASET_DIR}' 구조가 초기화되었습니다. 각 폴더에 학습 이미지를 넣고 다시 가동하십시오.")
    exit()

# ==========================================
# 1. 멀티태스크 커스텀 데이터 레이어 설계
# ==========================================
class EcoVisionMultiTaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_folder = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.materials = TARGET_MATERIALS
        self.objects = TARGET_OBJECTS

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img_path, mat_idx = self.image_folder.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        
        # [핵심] 파일명 텍스트 마이닝을 통한 물품(Object) 레이블 자동 매칭
        filename = os.path.basename(img_path)
        obj_idx = len(self.objects) - 1  # 기본값: '기타물품'
        
        for i, obj in enumerate(self.objects):
            if obj in filename:
                obj_idx = i
                break
        else:
            # 파일명에 명시적 이름이 없을 경우 재질 기반 논리적 자동 매칭 (폴백 방어)
            mat_name = self.materials[mat_idx]
            if mat_name == "plastic": obj_idx = 1     # 페트병
            elif mat_name == "paper": obj_idx = 2    # 종이컵
            elif mat_name == "metal": obj_idx = 3    # 음료수캔
            elif mat_name == "cardboard": obj_idx = 4 # 골판지상자
            elif mat_name == "trash": obj_idx = 5     # 일반비닐

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(obj_idx, dtype=torch.long), torch.tensor(mat_idx, dtype=torch.long)

# ==========================================
# 2. 이미지 증강 및 데이터 로더 파이프라인
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = EcoVisionMultiTaskDataset(root_dir=DATASET_DIR, transform=train_transform)
if len(full_dataset) == 0:
    print("❌ 학습 데이터셋이 비어 있습니다. dataset/ 하위 폴더에 이미지를 추가해 주세요.")
    exit()

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ==========================================
# 3. 멀티태스크 연산 모델 구조 정의
# ==========================================
class EcovisionMultiTaskModel(nn.Module):
    def __init__(self, num_objects, num_materials):
        super(EcovisionMultiTaskModel, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity() 
        
        self.object_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_objects)
        )
        self.material_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_materials)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.object_head(features), self.material_head(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EcovisionMultiTaskModel(num_objects=len(TARGET_OBJECTS), num_materials=len(TARGET_MATERIALS)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ==========================================
# 4. 고속 학습 및 검증 루프
# ==========================================
EPOCHS = 5
print(f"🚀 [인프라 가동] {device} 환경에서 멀티태스크 신경망 가중치 최적화를 시작합니다.")

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, obj_labels, mat_labels in train_loader:
        images, obj_labels, mat_labels = images.to(device), obj_labels.to(device), mat_labels.to(device)
        
        optimizer.zero_grad()
        obj_preds, mat_preds = model(images)
        
        # 두 가지 손실 함수 계수를 결합하여 역전파(Backpropagation) 수행
        loss_obj = criterion(obj_preds, obj_labels)
        loss_mat = criterion(mat_preds, mat_labels)
        loss = loss_obj + loss_mat
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - 통합 Loss: {epoch_loss:.4f}")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_NAME)

print(f"✅ [학습 완료] 최적화된 행렬 가중치가 '{MODEL_NAME}'으로 무결하게 저장되었습니다.")
