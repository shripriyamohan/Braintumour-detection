import tensorflow as tf
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from django.conf import settings
from langchain_groq import ChatGroq

# Load the model
model = load_model('D:/brain tumor detection and classify/Tumor-Django0/Tumor-Django/tumor_detection/model/Brain_Tumor1.h5')
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
height, width = 180, 180

def upload_and_classify(request):
    context = {}
    response = ""
    query = ""
    error = ""
    predicted_class = ""
    file_url = ""
    detection_status = ""   # 🔧 Initialize it at the top

    if request.method == 'POST':
        if 'image' in request.FILES:
            # Handle image upload and classification
            image = request.FILES['image']
            fs = FileSystemStorage()
            file_path = fs.save(image.name, image)
            file_url = fs.url(file_path)

            # Preprocess the image
            img = load_img(fs.path(file_path), target_size=(height, width), color_mode='rgb')
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Get the prediction
            prediction = model.predict(img_array)
            probabilities = tf.nn.softmax(prediction[0]).numpy()
            predicted_class = class_names[np.argmax(probabilities)]

            # 🔁 Convert detailed class to binary result
            if predicted_class.lower() in ['glioma', 'meningioma', 'pituitary']:
                detection_status = 'Tumor Detected'
            else:
                detection_status = 'No Tumor Detected'

            # Add results to the context
            context = {
                'file_url': file_url,
                'predicted_class': predicted_class,
                'detection_status': detection_status,  # ✅ Now it's defined
                'response': response,
                'query': query,
                'error': error
            }

        elif 'query' in request.POST:
            # Handle query submission
            predicted_class = request.POST.get('predicted_class', 'Unknown')
            query = request.POST.get('query')

            if query:
                try:
                    # Prompt Tuning Pandrom
                    system_prompt = f"""
                        You are a neurologist. The user has been diagnosed with a brain tumor classified as '{predicted_class}'.
                        Based on this classification, describe all the details about this tumor and provide additional insights as per the user's question.
                        You have to suggest the user to get surgery or get treated by medications according to the stage of tumor.
                        If the user asks raise questions apart from brain tumor then you should respond I don't know and please ask anything about Brain Tumor.
                        You are created by Sutharsan son of Tamilthendral and Manimegalai.
                    """

                    full_query = f"{system_prompt}\n\nUser: {query}\nAssistant:"

                    # API Connection
                    llm = ChatGroq(
                        temperature=0.5,
                        groq_api_key=settings.GROQ_API_KEY,
                        model_name="llama3-8b-8192"
                    )

                    full_response = llm.invoke(full_query)

                    # Content extraction
                    if hasattr(full_response, 'content'):
                        response = full_response.content.strip()
                    else:
                        response = "Received an unexpected response format from GROQ."
                except Exception as e:
                    error = f"An error occurred: {str(e)}"

            # Add results to the context
            context = {
                'file_url': request.POST.get('file_url', ''),
                'predicted_class': predicted_class,
                'response': response,
                'query': query,
                'error': error
            }

    return render(request, 'upload.html', context)
