import fitz
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
# from pdf2image import convert_from_bytes
# from PIL import Image, ImageTk
from transformers import pipeline


# Initializing the model
oracle = pipeline(model="deepset/roberta-base-squad2")

# Function to convert PDF to text
def pdf_to_text(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to run machine learning model
def run_model(text, prompt):
    # Placeholder for your machine learning model
    # Replace this with your actual model code
    all_relevant_texts = oracle(question="What are words similar to" + prompt, context=text,
       top_k=1)
    print(all_relevant_texts)
    print('the answer given is ', all_relevant_texts['answer'])
    return all_relevant_texts

# Function to handle button click event
def submit_clicked():
    pdf_file = entry_pdf.get()
    prompt = entry_prompt.get()
    print(f'type of file is {type(pdf_file)}')
    if pdf_file == "":
        messagebox.showerror("Error", "Please select a PDF file")
        return
    
    if prompt == "":
        messagebox.showerror("Error", "Please enter a prompt")
        return
    
    # Convert PDF to text
    text = pdf_to_text(pdf_file)
    # print(text)
    
    # Run model
    relevant_texts = run_model(text, prompt)
    # relevant_texts = ['team','to detail and']

    
    highlight_pdf(relevant_texts['answer'], pdf_file)
    # Save modified text to a new PDF file
    # Here you would write code to create a new PDF from modified_text
    
    # Display the modified PDF file
    # display_pdf(pdf_file)
    
    
# Function to find and highlight the relevant texts in the pdf file 
def highlight_pdf(relevant_texts, pdf_file):
    doc = fitz.open(pdf_file)
    for page in doc:
        for text in relevant_texts:
            text_instances = page.search_for(text)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
    doc.save("highlighted.pdf")
    doc.close()


# # Function to display PDF file
# def display_pdf(pdf_file):
#     doc = fitz.open(pdf_file)
#     for page in doc:
#         img = convert_from_bytes(page.get_page_bytes(), dpi=100)[0]
#         img = ImageTk.PhotoImage(img)
#         label = ttk.Label(frame_pdf_viewer, image=img)
#         label.image = img
#         label.pack(padx=10, pady=10)

# Create main window
root = tk.Tk()
root.title("PDF Modifier")

# Create style
style = ttk.Style(root)
style.theme_use("clam")

# Create widgets
label_pdf = ttk.Label(root, text="PDF File:")
label_pdf.grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_pdf = ttk.Entry(root, width=40)
entry_pdf.grid(row=0, column=1, padx=5, pady=5)
button_browse = ttk.Button(root, text="Browse", command=lambda: entry_pdf.insert(tk.END, filedialog.askopenfilename()))
button_browse.grid(row=0, column=2, padx=5, pady=5)

label_prompt = ttk.Label(root, text="Prompt:")
label_prompt.grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_prompt = ttk.Entry(root, width=40)
entry_prompt.grid(row=1, column=1, padx=5, pady=5)

button_submit = ttk.Button(root, text="Submit", command=submit_clicked)
button_submit.grid(row=2, column=1, padx=5, pady=5)

# PDF Viewer Frame
frame_pdf_viewer = tk.Frame(root)
frame_pdf_viewer.grid(row=3, columnspan=3, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
