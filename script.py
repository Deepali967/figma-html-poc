import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import openai
import matplotlib.pyplot as plt

# Load pre-trained model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/mobilenet_v2_100_224/1"  # Change if you prefer another model
model = hub.load(MODEL_URL)

# Function to run object detection using TensorFlow
def run_object_detection(image_path):
    # Load image using OpenCV
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Convert image into tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run the detection model
    detections = model(input_tensor)

    return detections, image_np

# Function to generate a prompt for OpenAI API based on detected UI components
def generate_prompt(detections, image_np):
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # Set a threshold to ignore low-confidence detections
    threshold = 0.5

    ui_elements = []

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image_np.shape[1])  # Convert normalized coordinates to pixels
            ymin = int(ymin * image_np.shape[0])
            xmax = int(xmax * image_np.shape[1])
            ymax = int(ymax * image_np.shape[0])

            # Map class IDs to UI component labels
            class_to_label = {
                1: "button",   # Example class ID mapping
                2: "input",    # Example class ID mapping
                3: "div",      # Example class ID mapping
                4: "header",   # Example class ID mapping
                5: "paragraph" # Example class ID mapping
            }

            tag = class_to_label.get(class_ids[i], "div")
            ui_elements.append({
                'tag': tag,
                'x': xmin,
                'y': ymin,
                'width': xmax - xmin,
                'height': ymax - ymin,
                'confidence': scores[i]
            })

    # Prepare a textual prompt for OpenAI based on detected UI elements
    prompt = "Generate HTML and CSS based on the following UI layout components. For each component, include the tag, position, and dimensions:\n"

    for element in ui_elements:
        prompt += f"Component: <{element['tag']}> | Position: ({element['x']}, {element['y']}) | Size: ({element['width']}x{element['height']}) | Confidence: {element['confidence']:.2f}\n"
    
    prompt += "\nGenerate HTML with appropriate tags and CSS for layout."

    return prompt

# Function to call OpenAI API and generate HTML/CSS
def generate_html_css_from_prompt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates HTML and CSS from UI descriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,  # Adjust based on the complexity of the generated content
        temperature=0.7
    )

    # Extract the response from OpenAI API
    generated_html_css = response['choices'][0]['text'].strip()
    return generated_html_css

# Function to display detected UI components with bounding boxes
def display_detected_objects(detections, image_np):
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # Set the threshold for detection (e.g., only display detections with score > 0.5)
    threshold = 0.5

    # Draw bounding boxes and labels on the image
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * image_np.shape[1])
            ymin = int(ymin * image_np.shape[0])
            xmax = int(xmax * image_np.shape[1])
            ymax = int(ymax * image_np.shape[0])

            # Draw the bounding box (green color)
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw the label and score
            label = f"ID {class_ids[i]}: {scores[i]:.2f}"
            cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert the image back to BGR for displaying using OpenCV/Matplotlib
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Display the result using Matplotlib
    plt.imshow(image_np)
    plt.axis('off')  # Hide axes for better display
    plt.show()

# Main function to run the object detection and HTML/CSS generation
def main(image_path):
    # Run object detection on the image
    detections, image_np = run_object_detection(image_path)

    # Display the detected objects
    display_detected_objects(detections, image_np)

    # Generate a prompt for OpenAI API based on detected UI components
    prompt = generate_prompt(detections, image_np)
    print(prompt);

    # Generate HTML and CSS using OpenAI API
    generated_html_css = generate_html_css_from_prompt(prompt)

    # Print the generated HTML and CSS
    print(generated_html_css)

# Set the image path (replace with your own image file path)
image_path = "sample.png"  # Replace with your image file path

# Run the main function
main(image_path)
