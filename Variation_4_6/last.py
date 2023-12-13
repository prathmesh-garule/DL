import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from swin_transformer_pytorch import SwinTransformer

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_data = ImageFolder('train', transform=transform)
dev_data = ImageFolder('dev', transform=transform)
test_data = ImageFolder('test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Swin Transformer model for transfer learning
model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3, num_classes=2, head_dim=32,
               window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # You can change this
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Calculate training loss
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# Evaluation function
def evaluate(loader):
    model.eval()
    predicted_probs = []
    true_labels = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Probability for class 1
            true_labels.extend(labels.cpu().numpy())
    
    return true_labels, predicted_probs

# Evaluate on dev set
dev_true_labels, dev_predicted_probs = evaluate(dev_loader)

# Calculate accuracy
dev_predicted = [1 if p >= 0.5 else 0 for p in dev_predicted_probs]
dev_accuracy = accuracy_score(dev_true_labels, dev_predicted)
print(f"Accuracy on Dev Set: {dev_accuracy:.2%}")

# Calculate and plot confusion matrix
dev_conf_matrix = confusion_matrix(dev_true_labels, dev_predicted)
plt.figure(figsize=(8, 6))
# Plot confusion matrix using heatmap (use seaborn or matplotlib)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(dev_true_labels, dev_predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the model
torch.save(model.state_dict(), '')

# Save the output CSV file for each image with its prediction information
dev_predictions_df = pd.DataFrame({
    'Image Name': dev_data.imgs,
    'True Value': dev_true_labels,
    'Predicted Value': dev_predicted,
    'Probability for Real': [1 - p for p in dev_predicted_probs],  # Probability for real class (class 0)
    'Probability for Fake': dev_predicted_probs  # Probability for fake class (class 1)
})
dev_predictions_df.to_csv('predictions_dev.csv', index=False)
