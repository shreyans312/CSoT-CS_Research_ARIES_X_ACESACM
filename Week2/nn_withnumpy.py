import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

train_images, train_labels = next(iter(train_loader))
test_images, test_labels = next(iter(test_loader))

x_train = train_images.view(-1, 28*28).numpy()
y_train = train_labels.numpy()

x_test = test_images.view(-1, 28*28).numpy()
y_test = test_labels.numpy()

x_train = x_train / 255
x_test = x_test / 255
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

input_dim = 28*28
output_dim = 10
lr = 0.01
epochs = 40
batch_size = 64
lambda_ = 1e-4

W = np.random.randn(input_dim, output_dim)*0.01
b = np.zeros((1, output_dim))

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def softmax(logits):
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    log_probs = -np.log(predictions[range(m), labels] + 1e-9)
    return np.mean(log_probs)

def accuracy(predictions, labels):
    pred_classes = np.argmax(predictions, axis=1)
    return np.mean(pred_classes == labels)

for epoch in range(epochs):
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, len(x_train), batch_size):
        X_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        logits = np.dot(X_batch, W) + b
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)
        loss += lambda_*np.sum(W*W)

        m = X_batch.shape[0]
        dlogits = probs
        dlogits[range(m), y_batch] -= 1
        dlogits /= m

        dW = np.dot(X_batch.T, dlogits) + 2*lambda_*W
        db = np.sum(dlogits, axis=0)
        W -= lr * dW
        b -= lr * db

    train_logits = np.dot(x_train, W) + b
    train_probs = softmax(train_logits)
    train_loss = cross_entropy_loss(train_probs, y_train)
    train_accuracy = accuracy(train_probs, y_train)

    test_logits = np.dot(x_test, W) + b
    test_probs = softmax(test_logits)
    test_loss = cross_entropy_loss(test_probs, y_test) + lambda_ * np.sum(W * W)
    test_acc = accuracy(test_probs, y_test)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy*100:.2f}%")

test_logits = np.dot(x_test, W) + b
test_probs = softmax(test_logits)
test_acc = accuracy(test_probs, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


test_pred_classes = np.argmax(test_probs, axis=1)
cm = confusion_matrix(y_test, test_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Test Set")
plt.show()
