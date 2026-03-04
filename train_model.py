import os
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from model_architecture import create_cnn_model, print_model_summary
from utils import (preprocess_mnist_data, plot_sample_images, plot_training_history, evaluate_model)


def main():
    
    
    # MNIST e construit in Keras
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()


    x_train = x_train_full[:50000]
    y_train = y_train_full[:50000]
    x_val = x_train_full[50000:]
    y_val = y_train_full[50000:]
    
    print(f"Setul de train {len(x_train)} imagini\n")
    print(f"Setul de validare {len(x_val)} imagini\n")
    print(f"Setul de testare {len(x_test)} imagini\n")

    print("Procesam datele...")
    
    x_train, y_train, x_val, y_val = preprocess_mnist_data(
        x_train, y_train, x_val, y_val
    )
    x_test_processed, y_test_processed, _, _ = preprocess_mnist_data(
        x_test, y_test, x_test[:1], y_test[:1]
    )
    
    plot_sample_images(x_train, y_train, num_samples=10)

    
    model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    print_model_summary(model)
    
    # Create directory for saving models
    os.makedirs('saved_models', exist_ok=True)
    
    callbacks = [
        # Opreste trainingul daca nu se schimba loss in 5 epoci
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducem rata de invatare daca loss stagneaza
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Salveaza modelul in timpul trainingul
        ModelCheckpoint(
            'saved_models/best_model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=128,  # Process 128 images at a time
        epochs=20,  # Maximum 20 epochs (early stopping may end sooner)
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1  # Show progress bar
    )

    print("Training complet\n")
    
    test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_processed, verbose=0)
    
    
    print(f"Rezultate\n")
    print(f"Acuratete: {test_accuracy * 100:.2f}%")
    print(f"Pierderi: {test_loss:.4f}")
    print(f"{'='*70}\n")
    
    # Evaluare
    evaluate_model(model, x_test_processed, y_test_processed)

    plot_training_history(history)
    
    model.save('saved_models/mnist_cnn_model.h5')
    print("Modelul salvat ca 'saved_models/mnist_cnn_model.h5'\n")
    
    sample_image = x_test_processed[0:1]
    sample_label = np.argmax(y_test_processed[0])
    
    prediction = model.predict(sample_image, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    
    print(f"True label: {sample_label}")
    print(f"Predicted: {predicted_digit} (Confidence: {confidence:.2f}%)")
    
    if predicted_digit == sample_label:
        print("Correct prediction!\n")
    else:
        print("Incorrect prediction\n")
    
    print("TRAINING PIPELINE COMPLETE")

if __name__ == "__main__":
    main()