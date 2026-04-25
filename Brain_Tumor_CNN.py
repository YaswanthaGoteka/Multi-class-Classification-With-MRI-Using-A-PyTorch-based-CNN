# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Data Augmentation
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # MRI is grayscale
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),            # augmentation
    transforms.RandomRotation(10),                # augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])            # grayscale normalization
])

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dara Loaders
train_ds = datasets.ImageFolder('data/Training', transform=train_tf)
test_ds  = datasets.ImageFolder('data/Testing', transform=test_tf)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_ds.classes
class_names.remove("notumor")
class_names.append("no tumor")
print("Classes:", class_names)


# Handles Class Imbalance
class_counts = [0] * len(class_names)

for _, label in train_ds:
    class_counts[label] += 1

print("Class counts:", class_counts)

weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()  # normalize
weights = weights.to(device)


# Model With Grayscale Input
model = nn.Sequential(
    nn.Conv2d(1,  32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32,  64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64,  128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, len(class_names)) # 4 classes
).to(device)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

model.train()

EPOCHS = 50

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)
        loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")

model.eval()
test_loss, correct = 0.0, 0

with torch.no_grad():
  for x, y in test_dl:
    x, y = x.to(device), y.to(device)

    logits = model(x)
    test_loss += loss_fn(logits, y).item() * y.size(0)

    preds = logits.argmax(dim=1)
    correct += (preds == y).sum().item()

test_loss /= len(test_dl.dataset)
accuracy = 100.0 * correct / len(test_dl.dataset)

print(f"Test Loss: {test_loss:.2f}       Test Accuracy {accuracy:.2f} %")

import random
from torchvision.transforms.functional import to_pil_image

correct = 0
incorrect = 0
total = 0
for i in range(1):
    index = random.randrange(len(test_ds))
    img, label = test_ds[index]

    # unnormalize
    unnorm = img * 0.5 + 0.5

    plt.imshow(to_pil_image(unnorm), cmap='gray')
    plt.title("Sample MRI")
    plt.axis('off')
    plt.show()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1).item()

    print("--- Prediction Breakdown ---")
    for i, p in enumerate(probs[0]):
        print(f"{class_names[i]}: {p.item()*100:.2f}%")

    print("Prediction:", class_names[pred])
    print("Actual:", class_names[label])

    if class_names[pred] == class_names[label]:
        correct += 1
        total += 1
    else:
        incorrect += 1
        total += 1
print(f"\nProportion of correct predictions of sample: {(correct/total):.2f}")
print(f"Number of incorrect predictions of sample: {incorrect}")

# Predicting Custom MRI Image
from PIL import Image

def predict_image(image_path, model, device):
    img = Image.open(image_path)

    if img.mode != 'L':
        img = img.convert('L')  # convert to grayscale

    transform = test_tf

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(1).item()
        confidence = probs[0][pred_idx].item()

    return pred_idx, confidence, img, probs

# Example usage
image_path = "test_mri.jpg"
pred_idx, confidence, img, probs = predict_image(image_path, model, device)

print("\nFinal Prediction:")
for i, p in enumerate(probs[0]):
    print(f"{class_names[i]}: {p.item()*100:.2f}%")

print(f"Prediction: {class_names[pred_idx]} ({confidence*100:.2f}%)")

plt.imshow(img, cmap='gray')
plt.title(f"Prediction: {class_names[pred_idx]} ({confidence*100:.1f}%)")
plt.axis('off')
plt.show()
