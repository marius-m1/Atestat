"""
Gradio Web Interface for MNIST Digit Recognition

This application provides an interactive web interface where users can:
1. Upload images of handwritten digits
2. Draw digits directly on a canvas
3. Get real-time predictions with confidence scores
4. View probability distributions for all digits
"""

import os
import numpy as np
import gradio as gr
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

from utils import preprocess_uploaded_image, predict_digit


# ========================================
# Load Pre-trained Model
# ========================================
MODEL_PATH = 'saved_models/mnist_cnn_model.h5'

def load_model():
    """Load the trained CNN model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Please run 'python train_model.py' first to train the model."
        )
    
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
    return model

# Load model at startup
model = load_model()


# ========================================
# Prediction Functions
# ========================================
def predict_from_upload(image):
    """
    Predict digit from uploaded image.
    
    Args:
        image: PIL Image or numpy array from Gradio
    
    Returns:
        tuple: (result_text, probability_chart)
    """
    if image is None:
        return "⚠️ Please upload an image first!", None
    
    try:
        # Preprocess the image
        processed_image = preprocess_uploaded_image(image)
        
        # Make prediction
        predicted_digit, confidence, probabilities = predict_digit(model, processed_image)
        
        # Create result text
        result_text = f"""
## 🎯 Prediction Result

### Recognized Number: **{predicted_digit}**

**Confidence:** {confidence:.2f}%

---

### Top 3 Predictions:
"""
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_3_indices:
            result_text += f"\n- **Digit {idx}**: {probabilities[idx]*100:.2f}%"
        
        # Create probability distribution chart
        fig = create_probability_chart(probabilities, predicted_digit)
        
        return result_text, fig
        
    except Exception as e:
        return f"❌ Error processing image: {str(e)}", None


def predict_from_drawing(canvas_data):
    """
    Predict digit from drawn canvas.
    
    Args:
        canvas_data: Dictionary with 'composite' image from Gradio Sketchpad
    
    Returns:
        tuple: (result_text, probability_chart)
    """
    if canvas_data is None:
        return "⚠️ Please draw a digit first!", None
    
    try:
        # Extract the image from canvas data
        if isinstance(canvas_data, dict) and 'composite' in canvas_data:
            image = canvas_data['composite']
        else:
            image = canvas_data
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Check if image is blank
        img_array = np.array(image.convert('L'))
        if np.max(img_array) < 10:  # Nearly all black
            return "⚠️ Canvas appears blank. Please draw a digit!", None
        
        # Preprocess and predict
        processed_image = preprocess_uploaded_image(image)
        predicted_digit, confidence, probabilities = predict_digit(model, processed_image)
        
        # Create result text
        result_text = f"""
## 🎯 Prediction Result

### Recognized Number: **{predicted_digit}**

**Confidence:** {confidence:.2f}%

---

### Top 3 Predictions:
"""
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_3_indices:
            result_text += f"\n- **Digit {idx}**: {probabilities[idx]*100:.2f}%"
        
        # Create probability chart
        fig = create_probability_chart(probabilities, predicted_digit)
        
        return result_text, fig
        
    except Exception as e:
        return f"❌ Error processing drawing: {str(e)}", None


def create_probability_chart(probabilities, predicted_digit):
    """
    Create a bar chart showing prediction probabilities for all digits.
    
    Args:
        probabilities: Array of probabilities for each digit (0-9)
        predicted_digit: The digit with highest probability
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    digits = list(range(10))
    colors = ['#2ecc71' if i == predicted_digit else '#3498db' for i in digits]
    
    bars = ax.bar(digits, probabilities * 100, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Digit', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('Prediction Confidence for Each Digit', fontsize=16, fontweight='bold')
    ax.set_xticks(digits)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# ========================================
# Gradio Interface
# ========================================
def create_interface():
    """
    Create the Gradio web interface with two tabs:
    1. Upload Image tab
    2. Draw Digit tab
    """
    
    with gr.Blocks(title="Handwritten Digit Recognition") as demo:
        
        # Header
        gr.Markdown(
            """
            # 🧠 Handwritten Digit Recognition with CNN
            
            This application uses a **Convolutional Neural Network** trained on the MNIST dataset 
            to recognize handwritten digits (0-9). The model achieves **~99% accuracy** on test data!
            
            ### How to use:
            - **Tab 1:** Upload an image of a handwritten digit
            - **Tab 2:** Draw a digit directly on the canvas
            
            ---
            """
        )
        
        # Create tabs
        with gr.Tabs():
            
            # ========================================
            # TAB 1: Upload Image
            # ========================================
            with gr.Tab("📤 Upload Image"):
                gr.Markdown("### Upload an image of a handwritten digit")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=300
                        )
                        upload_button = gr.Button("🔍 Recognize Digit", variant="primary", size="lg")
                        gr.Markdown(
                            """
                            **Tips for best results:**
                            - Use clear, high-contrast images
                            - Digit should be centered
                            - Dark digit on light background works best
                            """
                        )
                    
                    with gr.Column(scale=1):
                        upload_output_text = gr.Markdown(label="Prediction Result")
                        upload_output_plot = gr.Plot(label="Probability Distribution")
                
                upload_button.click(
                    fn=predict_from_upload,
                    inputs=upload_input,
                    outputs=[upload_output_text, upload_output_plot]
                )
            
            # ========================================
            # TAB 2: Draw Digit
            # ========================================
            with gr.Tab("✏️ Draw Digit"):
                gr.Markdown("### Draw a digit (0-9) on the canvas below")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        canvas_input = gr.Sketchpad(
                            type="pil",
                            label="Draw Here",
                            height=400,
                            width=400
                        )
                        with gr.Row():
                            draw_button = gr.Button("🔍 Recognize Digit", variant="primary", size="lg")
                            clear_button = gr.Button("🗑️ Clear Canvas", size="lg")
                        
                        gr.Markdown(
                            """
                            **Drawing tips:**
                            - Draw thick, bold digits
                            - Center your digit in the canvas
                            - Make digits large and clear
                            """
                        )
                    
                    with gr.Column(scale=1):
                        draw_output_text = gr.Markdown(label="Prediction Result")
                        draw_output_plot = gr.Plot(label="Probability Distribution")
                
                draw_button.click(
                    fn=predict_from_drawing,
                    inputs=canvas_input,
                    outputs=[draw_output_text, draw_output_plot]
                )
                
                clear_button.click(
                    fn=lambda: None,
                    outputs=canvas_input
                )
        
        # Footer
        gr.Markdown(
            """
            ---
            
            ### 📚 About the Model
            
            This CNN architecture includes:
            - **3 Convolutional layers** (32, 64, 128 filters) for feature extraction
            - **MaxPooling layers** to reduce spatial dimensions
            - **Batch Normalization** for stable training
            - **Dropout layers** to prevent overfitting
            - **Dense layers** for final classification
            
            **Dataset:** MNIST (70,000 handwritten digit images)
            
            ---
            
            Built with ❤️ using TensorFlow, Keras, and Gradio
            """
        )
    
    return demo


# ========================================
# Launch Application
# ========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("LAUNCHING GRADIO WEB INTERFACE")
    print("="*70 + "\n")
    
    interface = create_interface()
    
    # Launch with multiple fallback options
    try:
        print("🚀 Starting Gradio server...")
        print("📍 Trying to open browser automatically...")
        print("\n" + "-"*70)
        
        interface.launch(
            share=False,  # Set to True to create a public link
            server_name="0.0.0.0",  # Changed to allow all interfaces
            server_port=7860,
            show_error=True,
            inbrowser=True,  # Force browser to open
            quiet=False  # Show all output
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n⚠️  Port 7860 is already in use!")
            print("🔄 Trying alternative port 7861...")
            interface.launch(
                share=False,
                server_name="0.0.0.0",
                server_port=7861,
                show_error=True,
                inbrowser=True,
                quiet=False
            )
        else:
            raise e