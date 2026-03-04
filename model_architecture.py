from tensorflow import keras
from keras import layers, models


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                      input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='bn1'),  # Stabilize training
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),  # Reduce size by half
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        
        layers.Flatten(name='flatten'),
        
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout1'),
        
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),  
        
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',  
        loss='categorical_crossentropy',  
        metrics=['accuracy'] 
    )
    
    return model


def print_model_summary(model):
    
    print("\nMODEL ARCHITECTURE SUMMARY\n")
   
    model.summary()
    print(f"Parametri totali: {model.count_params():,}")
    


if __name__ == "__main__":
    model = create_cnn_model()
    print_model_summary(model)