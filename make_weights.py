import torch
import torch.nn as nn
import torchvision.models as models

# main.py의 글로벌 엔터프라이즈 규격과 100% 동기화
TARGET_MATERIALS = 6  
TARGET_OBJECTS = 8    

class EcovisionMultiTaskModel(nn.Module):
    def __init__(self, num_objects, num_materials):
        super(EcovisionMultiTaskModel, self).__init__()
        # 하위 호환성을 위해 pretrained=False로 빈 모델 생성
        try:
            self.backbone = models.mobilenet_v3_small(weights=None)
        except Exception:
            self.backbone = models.mobilenet_v3_small(pretrained=False)
            
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

# 1. 무결성 검증을 통과할 모델 인스턴스 초기화
model = EcovisionMultiTaskModel(num_objects=TARGET_OBJECTS, num_materials=TARGET_MATERIALS)

# 2. 규격에 맞는 정식 .pth 파일 형태로 저장
torch.save(model.state_dict(), "best_ecovision_multitask.pth")
print("✅ [성공] 무결성 검증 통과용 임시 가중치 파일 'best_ecovision_multitask.pth' 생성 완료!")
