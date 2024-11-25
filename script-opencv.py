import cv2
import pytesseract
import openai
import os

# OpenAI API Key
openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

def preprocess_image(image_path):
    """Preprocess the image to detect potential text regions."""
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to isolate text regions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return image, thresh

def detect_text_regions(image, binary_image):
    """Detect bounding boxes around potential text regions."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small regions (noise)
        if w > 30 and h > 10:  # Adjust thresholds as necessary
            # Crop the text region and store it
            region = image[y:y+h, x:x+w]
            regions.append((region, (x, y, w, h)))  # Also store the bounding box coordinates
    
    return regions, contours

def extract_text_from_image(region):
    """Extract text from the image region using pytesseract OCR."""
    text = pytesseract.image_to_string(region, config='--psm 6')
    return text.strip()

def create_prompt(image, regions):
    """Create a detailed prompt from the detected text regions."""
    prompt = "Generate an HTML and CSS layout based on the following design structure. Each region's details are as follows:\n\n"
    
    for i, (region, (x, y, w, h)) in enumerate(regions):
        # Extract text from the region
        text = extract_text_from_image(region)
        
        # Build the region description
        prompt += f"Region {i + 1}:\n"
        prompt += f"Position: ({x}, {y}), Width: {w}, Height: {h}\n"
        prompt += f"Text: '{text}'\n"
        prompt += "Description: This is a block of text that should be rendered within this region.\n"
        prompt += "-"*50 + "\n"
    
    return prompt

def send_to_openai(prompt):
    """Send the prompt to OpenAI's API to generate HTML and CSS."""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating HTML and CSS."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

def save_html_to_file(html_content):
    """Save the generated HTML to a file in the same folder as the script."""
    current_directory = os.path.dirname(os.path.realpath(__file__))  # Get the current folder of the script
    file_path = os.path.join(current_directory, "generated_page.html")

    with open(file_path, 'w') as file:
        file.write(html_content)
    
    return file_path

def main():
    # Path to the input image
    image_path = "path_to_your_image.jpg"  # Replace with your image path

    # Step 1: Preprocess the image
    image, binary_image = preprocess_image(image_path)

    # Step 2: Detect potential text regions and get contours
    text_regions, contours = detect_text_regions(image, binary_image)

    # Step 3: Create a prompt for OpenAI with all the detected text regions
    combined_prompt = create_prompt(image, text_regions)

    # Step 4: Send the combined prompt to OpenAI to generate HTML and CSS
    html_css = send_to_openai(combined_prompt)

    # Step 5: Save the generated HTML and CSS to a file
    file_path = save_html_to_file(html_css)

    print(f"Generated HTML file saved to: {file_path}")

if __name__ == "__main__":
    main()
