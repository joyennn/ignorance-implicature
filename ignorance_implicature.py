import os
import csv
import io
import time
import base64
from PIL import Image
from tqdm.auto import tqdm
import anthropic
import openai
import google.generativeai as genai
from requests.exceptions import ReadTimeout, ConnectTimeout
from google.api_core.exceptions import ServiceUnavailable





"""# Claude 3.5"""

# ─── 1) Clade model setup ──────────────────────────────

client = anthropic.Anthropic(
    api_key="your-api-key"
)


# ─── 2) Image preprocessing ────────────────────────────

def encode_image_base64(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def resize_and_encode_image_base64(path, size=(224, 224)):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def call_claude_with_image(image_path, prompt):
    image_base64 = resize_and_encode_image_base64(image_path)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()


# ─── 3) Data load ───────────────────────────────────────────

data = []
with open("data.csv", encoding="utf-8-sig") as rf:
    reader = csv.reader(rf)
    for row in reader:
        _id, image_path, obj, txt, context, modifier = row
        data.append([_id, image_path, obj, txt, context, modifier])

def resize_image(path: str, size=(224,224)) -> Image.Image:
    with Image.open(path) as img:
        return img.resize(size)


# ─── 4) Prompt generation ─────────────────────────────────────

def prompt1(obj, txt):
    return (
        "Is the following text appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Text: \"{txt}\""
    )

def prompt2_polar(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"Did you find four {obj} in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

def prompt2_howmany(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"How many {obj} did you find in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

prompt_variants = {
    "ex1_claude.csv":       prompt1,
    "ex2_polar_claude.csv": prompt2_polar,
    "ex2_howmany_claude.csv": prompt2_howmany,
}


# ─── 5) Scoring (5 x repetition) ─────────────────────────────────────

for filename, prompt_fn in prompt_variants.items():
    out_path = os.path.join(output_dir, filename)
    rows = [["id", "image_path", "text", "context", "modifier", "score"]]

    for row in tqdm(data, desc=f"Running {filename}", leave=False):
        _id, img_path, obj, txt, context, modifier = row

        for rep in range(5):
            prompt = prompt_fn(obj, txt)
            try:
                score = call_claude_with_image(img_path, prompt)
            except Exception as e:
                print(f"Error on {_id}: {e}")
                score = "ERROR"

            rows.append([_id, img_path, txt, context, modifier, score])


    with open(out_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)

    print(f"Saved to: {out_path}")





"""# GPT-4o"""

# ─── 1) GPT model setup ───────────────────────────

openai.api_key = "your-api-key"

def load_gpt4o():
    return openai.chat.completions

def analyze_gpt4o(img_b64: str, prompt: str) -> str:
    messages = [{
      "role":"user",
      "content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}}
      ]
    }]
    resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=60,
            temperature=0.0
        )

    return resp.choices[0].message.content.strip()


# ─── 2) Image preprocessing ─────────────────────────────────────────

def resize_image(path: str, size=(224,224)) -> Image.Image:
    with Image.open(path) as img:
        return img.resize(size)

def encode_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─── 3) Data load ───────────────────────────────────────────

data = []
with open("data.csv", encoding="utf-8-sig") as rf:
    reader = csv.reader(rf)
    for row in reader:
        _id, image_path, obj, txt, context, modifier = row
        data.append([_id, image_path, obj, txt, context, modifier])

preprocessed = {}
for path in tqdm({row[1] for row in data}, desc="Preprocess Images"):
    img_resized = resize_image(path)
    b64         = encode_base64(img_resized)
    preprocessed[path] = {"b64": b64}


# ─── 4) Prompt generation ─────────────────────────────────────

def prompt1(obj, txt):
    return (
        "Is the following text appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Text: \"{txt}\""
    )

def prompt2_polar(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"Did you find four {obj} in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

def prompt2_howmany(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"How many {obj} did you find in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

prompt_variants = {
    "ex1_gpt4o.csv":        prompt1,
    "ex2_polar_gpt4o.csv":  prompt2_polar,
    "ex2_howmany_gpt4o.csv":prompt2_howmany,
}



# ─── 5) Scoring (5 x repetition)─────────────────────────────

for filename, prompt_fn in prompt_variants.items():
    out_path = os.path.join(output_dir, filename)
    rows = [["id", "image_path", "text", "context", "modifier", "score"]]

    for row in tqdm(data, desc=f"Running {filename}", leave=False):
        _id, img_path, obj, txt, context, modifier = row
        b64 = preprocessed[img_path]["b64"]

        for rep in range(5):
            prompt = prompt_fn(obj, txt)
            score  = analyze_gpt4o(b64, prompt)
            rows.append([_id, img_path, txt, context, modifier, score])

    with open(out_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)

    print(f"Saved to Drive: {out_path}")





"""# Gemini 1.5 Pro"""

# ─── 1) Gemini model setup ───────────────────────────

import google.generativeai as genai

genai.configure(api_key="your-api-key")

def load_gemini():
    """
    Gemini 1.5 Pro 모델을 로드해서 반환합니다.
    """
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_api = load_gemini()

def retryable_analyze_gemini(img_b64: str, prompt: str,
                             max_retries: int = 3,
                             backoff_sec: float = 5.0) -> str:

    for attempt in range(1, max_retries+1):
        try:
            image_data = {"mime_type": "image/png", "data": img_b64}
            gen_config = genai.types.GenerationConfig(
                max_output_tokens=60,
                temperature=0.0
            )
            resp = gemini_api.generate_content(
                [prompt, image_data],
                generation_config=gen_config
            )
            return resp.text.strip()
        except (ReadTimeout, ConnectTimeout, ServiceUnavailable) as e:
            if attempt == max_retries:
                raise
            else:
                wait = backoff_sec * attempt
                print(f"[gemini] attempt {attempt} failed: {e!r}. retrying in {wait}s …")
                time.sleep(wait)


# ─── 2) Image preprocessing ─────────────────────────────────────────

def resize_image(path: str, size=(224,224)) -> Image.Image:
    with Image.open(path) as img:
        return img.resize(size)

def encode_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ─── 3) Data load ───────────────────────────────────────────
data = []
with open("data.csv", encoding="utf-8-sig") as rf:
    reader = csv.reader(rf)
    for row in reader:
        _id, image_path, obj, txt, context, modifier = row
        data.append([_id, image_path, obj, txt, context, modifier])

preprocessed = {}
for path in tqdm({row[1] for row in data}, desc="Preprocess Images"):
    img_resized = resize_image(path)
    b64         = encode_base64(img_resized)
    preprocessed[path] = {"b64": b64}

# ─── 4) Prompt generation ─────────────────────────────────────

def prompt1(obj, txt):
    return (
        "Is the following text appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Text: \"{txt}\""
    )

def prompt2_polar(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"Did you find four {obj} in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

def prompt2_howmany(obj, txt):
    return (
        "Is the following answer to the question appropriate for the given image?\n"
        "Please reply with a single integer between 1 and 7, where 1 means “not at all appropriate” and 7 means “completely appropriate.”\n"
        f"Question: \"How many {obj} did you find in the boxes?\"\n"
        f"Answer: \"{txt}\""
    )

prompt_variants = {
    "ex1_gemini.csv":        prompt1,
    "ex2_polar_gemini.csv":  prompt2_polar,
    "ex2_howmany_gemini.csv":prompt2_howmany,
}


# ─── 5) Scoring (5 x repetition)─────────────────────────────

for filename, prompt_fn in prompt_variants.items():
    out_path = os.path.join(output_dir, filename)
    rows = [["id", "image_path", "text", "context", "modifier", "score"]]

    for row in tqdm(data, desc=f"Running {filename}", leave=False):
        _id, img_path, obj, txt, context, modifier = row
        b64 = preprocessed[img_path]["b64"]

        for rep in range(5):
            prompt = prompt_fn(obj, txt)
            score  = retryable_analyze_gemini(b64, prompt)
            rows.append([_id, img_path, txt, context, modifier, score])

    with open(out_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerows(rows)

    print(f"Saved to Drive: {out_path}")
