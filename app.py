import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from langchain_groq import ChatGroq
from langchain_core.messages import *
from markdown_pdf import MarkdownPdf, Section
from gtts import gTTS
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

# Making Directory if it doesnt exist
if not os.path.exists("Images"):
    os.makedirs("Images")

if not os.path.exists("Audio"):
    os.makedirs("Audio")

def safe_filename(prompt: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:100]

#Loading Duffusion model for cerating Images 
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")

def generate_image(pipe, prompt: str, page: int) -> str:
    torch.cuda.empty_cache()
    with torch.inference_mode():
        image = pipe(prompt).images[0]

    filename = safe_filename(prompt) + f"_page{page}.png"
    filepath = os.path.join("Images", filename)
    image.save(filepath)
    torch.cuda.empty_cache()
    return filepath


system_prompt = """
You only need to provide JSON as the output that can be easily parsed by the Python json library using json.loads.
Do not include any additional text, explanations, or descriptions.
You are a highly skilled and creative story writer specializing in crafting original, immersive, and engaging stories. Your stories must incorporate vivid, sensory-rich imagery and deeply compelling characters to fully captivate the reader's imagination and emotions. Each story should transport readers to unique, richly described worlds using diverse literary devices such as metaphors, similes, personification, and other expressive techniques to enhance the narrative’s artistic depth and atmosphere.
The story length should be approximately 1000 words, thoughtfully divided into five substantial sections, each representing one page of content. Each section should be richly detailed with immersive descriptions of settings, nuanced character development, and intricately woven plot progression. Include natural and meaningful dialogues between characters to add realism and emotional resonance to the story. Ensure the narrative has a well-defined structure with a captivating beginning that sets the tone, a middle that builds tension and develops complexity, and a satisfying conclusion that resolves major story arcs for a fulfilling reader experience.
The story topic or theme will be provided by the user at runtime. Your writing style should be adaptable and flexible to fit varying genres and target audiences while maintaining creativity and engagement throughout.
Write a story of ~1000 words split into exactly 5 sections (like 5 pages). Each sectiion shoulkd contain only 200 words at max. Take inspiration from existing story books available online.
Each page should have 200 words at max . Make sure to write in a child story book way. Make it interesting for children.
For each section generate:
1. title (short heading)
2. content (~200 words)
3. image_prompt (Each image should realistically illustrate the scene of its section with detailed, high-quality visuals. Describe characters, settings, colors, lighting, and atmosphere so the model can create photorealistic, storybook-style illustrations. Draw inspiration from real-world photography and professional children’s book artwork to ensure the images look clear, natural, and visually appealing, avoiding distortions or abstract outputs.Cahracters should be consistent throught out the story)

Output format:
{
  "pages": [
    {"title": "Page 1 Title", "content": "Text...", "image_prompt": "Prompt..."},
    {"title": "Page 2 Title", "content": "Text...", "image_prompt": "Prompt..."},
    {"title": "Page 3 Title", "content": "Text...", "image_prompt": "Prompt..."},
    {"title": "Page 4 Title", "content": "Text...", "image_prompt": "Prompt..."},
    {"title": "Page 5 Title", "content": "Text...", "image_prompt": "Prompt..."}
  ]
}
"""
#Used GROQ API KEY.Enter your own key here.
def generate_story(topic: str):
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"The topic of the story is {topic}"),
    ]
    response2 = llm.invoke(messages)
    return response2

def generate_markdown(pages) -> str:
    content_with_images = ""
    for idx, page in enumerate(pages, 1):
        abs_path = os.path.abspath(page["image_path"]).replace("\\", "/")
        content_with_images += f"""# {page['title']}
{page['content']}

<p align="center">
  <img src="file://{abs_path}" width="400"/>
</p>

---
"""
    return content_with_images

def save_pdf(content_with_image: str, title: str) -> str:
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(content_with_image))
    filepath = f"{safe_filename(title)}.pdf"
    pdf.save(filepath)
    return filepath

#USed gTTS for text to speech 
def generate_audio(text: str, page: int) -> str:
    filename = f"Audio/page_{page}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# ---STREAMLIT for Frontend---
st.title("Magic Storybook ✨📕")
topic = st.text_input("Enter a topic for your story:")

if st.button("Generate your Storybook"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("✨ Generating your story..."):
            story = generate_story(topic)
            try:
                response_json = json.loads(story.content)
                pages = response_json["pages"]
            except:
                st.error("Failed to parse story JSON.")
                st.write("Response was:", story.content)
                st.stop()

        pipe = load_pipeline()
        for i, page in enumerate(pages, 1):
            with st.spinner(f"✨The fairy is painting your picture with magic dust…page{i}..."):
                image_path = generate_image(pipe, page["image_prompt"], i)
                page["image_path"] = image_path
            audio_path = generate_audio(page["content"], i)
            page["audio_path"] = audio_path

            with st.expander(f"📖 Page {i}: {page['title']}", expanded=(i == 1)):
                st.write(page["content"])
                st.image(page["image_path"], caption=f"Illustration - Page {i}", use_container_width=True)
                st.audio(page["audio_path"], format="audio/mp3")

        content_with_image = generate_markdown(pages)
        pdf_path = save_pdf(content_with_image, topic)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Full Storybook (PDF)",
                data=f,
                file_name=f"{safe_filename(topic)}.pdf",
                mime="application/pdf"
            )


