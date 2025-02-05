# EssenceExtractor: AI-Powered Podcast Summarizer

EssenceExtractor is an innovative tool that leverages advanced AI technologies to transcribe and summarize podcast audio and video files. By simply uploading your podcast content, you can receive concise summaries, enabling you to grasp key insights without listening to the entire episode.
![image](https://github.com/user-attachments/assets/57638d5b-52c6-4b04-9802-0307a871045e)

## Key Features

- **Efficient Transcription**: tilizes OpenAI's Whisper model to accurately transcribe audio content.- **Insightful Summaries**: mploys advanced language models to distill transcripts into concise bullet points.- **User-Friendly Interface**: uilt with Streamlit for a seamless and intuitive user experience.- **Image Generation (Upcoming)**: n upcoming update will introduce image generation capabilities, providing visual representations of your podcast content.
## Installation

To run EssenceExtractor locally, follow these steps:

1. **Clone the Repository**:

   ``bash
   git clone https://github.com/YourUsername/EssenceExtractor.git
   ```
2. **Navigate to the Project Directory**:

   ``bash
   cd EssenceExtractor
   ```
3. **Install Dependencies**:

   ``bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**:

   reate a `.env` file in the project root and add your API keys:
   ``
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
5. **Run the Application**:

   ``bash
   streamlit run app.py
   ```
## Usage

1. **Upload a Podcast File**: lick on the "Choose a file" button to upload your podcast audio or video file.2. **Transcription**: fter uploading, click the "Summarize" button to transcribe the audio content.3. **Summary Generation**: nce transcription is complete, the application will generate a concise summary of the podcast.4. **Image Generation (Upcoming)**: n the next update, you will be able to generate visual representations of your podcast content.
## Acknowledgments

pecial thanks to Muhammad Usama for his motivation and support in developing EssenceExtractor.
## License

this project is licensed under the MIT License.
For more information and updates, visit the [EssenceExtractor GitHub Repository](https://github.com/Jurik-001/EssenceExtractor). 
