import os
import tempfile
from flask import Flask, render_template, request, send_file, make_response, jsonify
import fitz
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Initializing the model
oracle = pipeline("question-answering", model="deepset/roberta-base-squad2")
models = [oracle]

file_name_counter = 0
temp_dir = None
temp_file_path = None
text = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Main function
@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global file_name_counter, temp_dir, temp_file_path, text
    pdf_file = request.files['pdf_file']
    prompt = request.form['prompt']

    if pdf_file and prompt:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, pdf_file.filename)
        pdf_file.save(temp_file_path)

        # Convert PDF to text
        text = pdf_to_text(temp_file_path)
        text = 'This is the resume of a person.' + text 
        modified_pdf_files = []

        # Doing prompt engineering
        prompts = [f'What {prompt} did he do?',
                   f'{prompt}?',
                #    f'Where did he do {prompt}?'
                   f'What are some things related to {prompt}?',
                   f'How did he show {prompt}?']

        # Run models and highlight relevant texts
        relevant_texts = []
        for prompt in prompts :
            model_returns = run_model(text, prompt, oracle)
            for answer in model_returns:
                phrases = answer['answer'].split('\n')
                for phrase in phrases:
                    if len(phrase) >= 3:
                        if phrase not in relevant_texts:
                            relevant_texts.append(phrase)
                    if len(relevant_texts)>15:
                        break
            if len(relevant_texts)>15:
                break
        # print(len(relevant_texts))


        print('Passing the relevant texts to highlight: ', relevant_texts)
        modified_pdf_file = f'Highlighted_{file_name_counter}.pdf'
        highlight_pdf(relevant_texts, temp_file_path, modified_pdf_file)
        modified_pdf_files.append(modified_pdf_file)
        file_name_counter += 1

        return jsonify({'modified_pdf_files': modified_pdf_files})

    return jsonify({'error': 'Invalid request'}), 400

# Handles the question/answer section
@app.route('/process_question', methods=['POST'])
def process_question():
    global text
    question = request.get_json().get('question')
    if question and text:
        # Run models and get the answer
        for model in models:
            answer = run_model(text, question, model)[0]['answer']

        return jsonify({'answer': answer})
    return jsonify({'error': 'Invalid request'}), 400

# Function to convert PDF to text
def pdf_to_text(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to find and highlight the relevant texts in the pdf file
def highlight_pdf(relevant_texts, pdf_file, modified_pdf_file):
    total = len(relevant_texts)/3
    doc = fitz.open(pdf_file)
    rgb = [(255, 196, 126),(255, 227, 130),(255, 247, 138)]
    colors = []
    for color in rgb:
        colors.append((color[0]/255,color[1]/255,color[2]/255))
    # colors= [(208/255, 72/255, 72/255),(243/255,185/255, 95/255),(253/255, 231/255, 103/255)]
    # fills = [(1, 0.9, 0.7),(1, 1, 0.8),(1, 1, 0.9)]
    for page_num in range(len(doc)):
        page = doc[page_num]
        page.clean_contents()
        for index,text in enumerate(relevant_texts):
            if text.isspace():
                continue
            if len(text) < 3:
                print('skipping')
                continue
            text_instances = page.search_for(text)
            for inst in text_instances:
                try:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=colors[int(index//total)])
                    highlight.update()
                    # page.draw_rect(inst, color=(1, 1, 0), fill=(1, 1, 0.9))
                    # page.draw_rect(inst, color=colors[int(index//total)], fill=fills[int(index//total)], width=1.5,overlay=False,stroke_opacity=0)
                except ValueError as e:
                    print(f"Error occurred while highlighting: for {text} {e}")
                    continue

    doc.save(modified_pdf_file)
    doc.close()

# Function to run the model
def run_model(text, prompt, model):

   # top_k : The model will return the best_k answers
   all_relevant_texts = model(question=prompt, context=text, top_k=10)
   return all_relevant_texts

# Takes the input pdf
@app.route('/get_pdf_data/<path:pdf_file>', methods=['GET'])
def get_pdf_data(pdf_file):
    try:
        with open(pdf_file, 'rb') as f:
            pdf_data = f.read()
        response = make_response(pdf_data)
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'inline; filename="%s"' % os.path.basename(pdf_file))
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)