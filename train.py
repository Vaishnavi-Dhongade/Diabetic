# Importing Libraries
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


# Model Definition

class RetinopathyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        
    def forward(self, x):
        return self.model(x)
    

# Training Function

def train_model():
    try:

        # Image Pre-processing
        print("Initializing training...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        

        # Dataset And DataLoader
        dataset = datasets.ImageFolder('dataset', transform=transform)
        if len(dataset) == 0:
            print("Error: No images found in dataset folder")
            return
        
        print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Device Configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model, Loss, and Optimizer Initialization
        model = RetinopathyModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training Loop
        best_acc = 0
        for epoch in range(20):
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/20'):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item()
                
            acc = 100 * correct / total
            print(f'\nEpoch {epoch+1}: Loss={running_loss/len(dataloader):.3f}, Accuracy={acc:.2f}%')
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'model/best_model.pth')
                print(f'Saved model with accuracy: {acc:.2f}%')
                
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == '__main__':
    train_model()
