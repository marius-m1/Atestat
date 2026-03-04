
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


def preprocess_mnist_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    from keras.utils import to_categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test


def plot_sample_images(x_data, y_data, num_samples=10):
   
    plt.figure(figsize=(15, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_data[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(y_data[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sample images saved as 'sample_images.png'")


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history saved as 'training_history.png'")


def evaluate_model(model, x_test, y_test):
    # Predictii
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Report
    print("CLASSIFICATION REPORT\n")
    print(classification_report(y_true_classes, y_pred_classes, 
                                target_names=[str(i) for i in range(10)]))
    
    # Matrice de confuzie
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Digit Recognition', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")
    print("="*70 + "\n")


def preprocess_uploaded_image(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('L')
    else:
        if isinstance(image_path_or_array, Image.Image):
            img = image_path_or_array.convert('L')
        else:
            img = Image.fromarray(image_path_or_array).convert('L')
    
    img = img.resize((28, 28), Image.LANCZOS)
    

    img_array = np.array(img)
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    

    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array


def predict_digit(model, image):
   
    predictions = model.predict(image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100
    
    return predicted_digit, confidence, predictions[0]