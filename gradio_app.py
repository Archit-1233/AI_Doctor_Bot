from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import datetime
from fpdf import FPDF

from backend.brain_of_the_doctor import encode_image, analyze_image_with_query
from backend.voice_of_the_patient import record_audio, transcribe_with_groq
from backend.voice_of_the_doctor import text_to_speech_with_gtts

# System prompt
system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# PDF Generator
def generate_pdf(transcription, diagnosis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Doctor Report")
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(200, 10, txt=f"AI Doctor Consultation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {date_str}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Patient's Description:\n{transcription}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Doctor's Diagnosis:\n{diagnosis}")
    report_path = "doctor_report.pdf"
    pdf.output(report_path)
    return report_path

# Main logic
def process_inputs(audio_filepath, image_filepath):
    api_key = os.environ.get('GROQ_API_KEY')
    speech_to_text_output = transcribe_with_groq(api_key=api_key,
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        doctor_response = "No image provided for me to analyze."

    voice_of_doctor = text_to_speech_with_gtts(input_text=doctor_response, output_filepath="final.mp3")
    pdf_path = generate_pdf(speech_to_text_output, doctor_response)

    return speech_to_text_output, doctor_response, "final.mp3", pdf_path

# Gradio Interface
with gr.Blocks(title="AI Doctor with Vision and Voice") as iface:
    gr.Markdown("""
    # 🩺 AI Doctor
    Welcome to your AI-powered consultation. Speak your symptoms and upload a medical image.
    The AI doctor will diagnose and advise. All responses are for **learning/demo purposes only**.
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak Your Symptoms")
            image_input = gr.Image(type="filepath", label="🖼️ Upload Medical Image")
            submit_btn = gr.Button("Analyze")

        with gr.Column():
            stt_output = gr.Textbox(label="📝 Transcribed Speech")
            doctor_response = gr.Textbox(label="💬 Doctor's Response")
            audio_output = gr.Audio(label="🔈 Doctor's Voice")
            pdf_output = gr.File(label="📄 Download Your Report (PDF)")

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[stt_output, doctor_response, audio_output, pdf_output]
    )

iface.launch(debug=True)
