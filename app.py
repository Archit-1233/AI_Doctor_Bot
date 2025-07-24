from flask import Flask, render_template, request, redirect, send_file, url_for
import os
from backend.brain_of_the_doctor import encode_image, analyze_image_with_query
from backend.voice_of_the_patient import transcribe_with_groq
from backend.voice_of_the_doctor import text_to_speech_with_gtts
from fpdf import FPDF
import datetime
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/assets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Doctor prompt
system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

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
    report_path = os.path.join(app.config['OUTPUT_FOLDER'], "doctor_report.pdf")
    pdf.output(report_path)
    return report_path

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']
        audio = request.files['audio']

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
        image.save(image_path)
        audio.save(audio_path)

        stt_output = transcribe_with_groq(api_key=os.getenv("GROQ_API_KEY"),
                                          stt_model="whisper-large-v3",
                                          audio_filepath=audio_path)

        if image:
            doctor_response = analyze_image_with_query(
                query=system_prompt + stt_output,
                encoded_image=encode_image(image_path),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            doctor_response = "No image provided for analysis."

        voice_path = os.path.join(app.config['OUTPUT_FOLDER'], "final.mp3")
        text_to_speech_with_gtts(input_text=doctor_response, output_filepath=voice_path)

        pdf_path = generate_pdf(stt_output, doctor_response)

        return render_template("result.html",
                               stt=stt_output,
                               diagnosis=doctor_response,
                               voice_url="/static/assets/final.mp3",
                               pdf_url="/static/assets/doctor_report.pdf")

    return render_template('upload.html')

@app.route('/download-report')
def download_report():
    return send_file('static/assets/doctor_report.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
