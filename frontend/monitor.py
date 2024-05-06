import os
import pandas as pd
import streamlit as st

from src.data_preprocessing import preprocess
from src.dataimporter import dataimporter
import tensorflow as tf


import shutil




def list_models(folder_path):
    return [model for model in os.listdir(folder_path) if model.endswith('.h5')]

def delete_model(folder_path, model_name):
    model_path = os.path.join(folder_path, model_name)
    if os.path.exists(model_path):
        # Ensure it's not the first model before deleting
        models = list_models(folder_path)
        if model_name != models[0]:
            os.remove(model_path)
            st.success(f"Model '{model_name}' deleted successfully.")
        else:
            st.warning(f"Model '{model_name}' is the default model and cannot be deleted.")
    else:
        st.error(f"Model '{model_name}' not found.")

def remove_non_default_models(folder_path):
    models = list_models(folder_path)
    
    if not models:
        st.error("No model files found in the folder.")
        return
    
    st.subheader("Models (Except First One):")
    non_default_models = models[1:]  # Skip the first model
    selected_model = st.radio("Select model to delete:", non_default_models)
    
    if st.button("Delete Model") and selected_model:
        delete_model(folder_path, selected_model)

# Main streamlit app
def new_delete():
    st.title("Model Removal Tool")
    folder_path = "./models"  # Absolute path to the models folder

    if not os.path.exists(folder_path):
        st.error(f"Error: Models folder not found at path {folder_path}.")
        return
    
    remove_non_default_models(folder_path)














def calculate_accuracy(model_path, test_ds):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Initialize variables for calculating accuracy and loss
    num_samples = 0
    correct_predictions = 0
    total_loss = 0.0
    
    # Iterate through the test dataset to evaluate the model
    for image_batch, labels_batch in test_ds:
        # Predict batch of images
        batch_predictions = model.predict(image_batch)
        # Compare predictions to actual labels
        batch_correct = tf.argmax(batch_predictions, axis=1) == tf.cast(labels_batch, tf.int64)
        # Count correct predictions in the batch
        correct_predictions += tf.reduce_sum(tf.cast(batch_correct, tf.int32)).numpy()
        # Update total number of samples
        num_samples += image_batch.shape[0]
        
        # Calculate batch loss
        batch_loss = tf.keras.losses.sparse_categorical_crossentropy(labels_batch, batch_predictions, from_logits=True)
        total_loss += tf.reduce_sum(batch_loss).numpy()
    
    # Calculate accuracy percentage
    accuracy_percentage = (correct_predictions / num_samples) * 100
    # Calculate average loss
    average_loss = total_loss / num_samples
    return round(accuracy_percentage, 2), round(average_loss, 4)



def load_models(models_folder, test_ds):
    models_info = []
    for model_file in os.listdir(models_folder):
        model_path = os.path.join(models_folder, model_file)
        # Load the model and calculate accuracy and loss on test dataset
        accuracy, loss = calculate_accuracy(model_path, test_ds)
        models_info.append({"Model Name": model_file, "Accuracy": accuracy, "Loss": loss})
    return pd.DataFrame(models_info)




def model_monitor(models_folder, test_ds):
    st.markdown(
        """
        <style>
        .title-text {
            color: #ffffff;
            background-image: linear-gradient(to right, #ff4d4d, #0066ff);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-family: Arial, sans-serif;
            margin-bottom: 20px;
        }
        </style>
         
        """,
        unsafe_allow_html=True)

    
    
    
    
    
    st.markdown("""<div class=title-text>
          Model Monitoring
         </div>""",unsafe_allow_html=True)
    
    # Load models and get accuracy info
    models_df = load_models(models_folder, test_ds)
    
    # Display accuracy info in a dataframe
    st.dataframe(models_df)
  
    # Create line chart for accuracy visualization
    st.write("accuracy graph: ")
    chart_data = models_df.set_index("Model Name")
    st.line_chart(chart_data["Accuracy"])
    st.write("Loss graph: ")
    chart_data = models_df.set_index("Model Name")
    st.line_chart(chart_data["Loss"])
    
    
    
def monitor():
    models_folder = "./models"
    dataset = dataimporter()
    train_ds, val_ds, test_ds = preprocess(dataset)  #test dataset
    
   
    
    
    
    # Apply styling
    st.markdown(
        """
        <style>
        .dataframe {
            border: 1px solid #cccccc;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .chart-container {
            border: 2px solid #cccccc;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display model monitoring content
    model_monitor(models_folder, test_ds)
    new_delete()
   

