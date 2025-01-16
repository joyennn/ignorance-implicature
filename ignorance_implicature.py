import csv

data = []
with open('data.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)



### GPT-4o ###
import openai
import base64
from PIL import Image

# OpenAI API key
openai.api_key = "your-api-key"

def resize_image(image_path, output_path, size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(size)
        img.save(output_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path, prompt):
    resized_path = "resized_image.png"
    resize_image(image_path, resized_path, size=(224, 224))
    base64_image = encode_image(resized_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=60
    )

    return response.choices[0].message.content


for i in data:
    image_path = i[1]
    text = i[2]
    prompt = f"Based on the given image, is the following text appropriate?: {text} Answer 'True' or 'False'"
    result = analyze_image(image_path, prompt)
    i.append(result)



### Gemini 1.5 pro ###
import google.generativeai as genai
import base64
from PIL import Image

def resize_image(image_path, output_path, size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.resize(size)
        img.save(output_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path, prompt):
    # Google API key
    api_key = "your-api-key"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    resized_path = "resized_image.png"
    resize_image(image_path, resized_path, size=(224, 224))
    encoded_image = encode_image(resized_path)

    image_data = {
        "mime_type": "image/png",
        "data": encoded_image}

    response = model.generate_content(
        [prompt, image_data],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=100
        )
    )

    return response.text


for i in data:
    image_path = i[1]
    text = i[2]
    prompt = f"Based on the given image, is the following text appropriate?: {text} Answer 'True' or 'False'"
    result = analyze_image(image_path, prompt)
    i.append(result)



# save the result
output_file = "result.csv"

with open(output_file, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)