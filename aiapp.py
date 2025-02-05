import streamlit as st
import whisper
from diffusers import AutoPipelineForText2Image
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

##pip install diffusers transformers accelerate --upgrade


st.set_page_config(
    page_title="Shees Pod",
   
)







# Set up Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

groq_api_key=os.environ['GROQ_API_KEY']
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="deepseek-r1-distill-llama-70b")

# Summarize text using Gemini
def summarize_text(text, llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following podcast transcript in 3-5 bullet points\n\n{text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary

# Main function
def main():
    load_dotenv()
    st.title("EssenceExtractor 🎙")
    st.write("Upload a podcast audio or video file to get a summary.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Summarize"):
            with st.spinner("Transcribing audio..."):
                # Save the uploaded file temporarily
                with open("temp_audio.mp3", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Transcribe using Whisper
                model = load_whisper_model()
                result = model.transcribe("temp_audio.mp3")
                transcription = result["text"]
                
                st.success("Transcription complete!")
                ##pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
                ##pipe.to("cuda")

                ##prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

                ##image = pipe(prompt=transcription, num_inference_steps=1, guidance_scale=0.0).images[0] 
            with st.spinner("Generating summary..."):
                # Summarize using Groq
               
                summary = summarize_text(transcription, llm)
                st.success("Summary generated!")

            # Display results
            st.subheader("Transcription:")
            st.write(transcription)

            # Display summary in the left panel
            st.sidebar.subheader("Summary:")
            st.sidebar.write(summary)

            # Clean up temporary file
            os.remove("temp_audio.mp3")

if __name__ == "__main__":
    main()