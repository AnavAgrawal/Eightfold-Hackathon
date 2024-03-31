# # from transformers import pipeline
# # import pytesseract

# # pytesseract.pytesseract.tesseract_cmd = "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# # document_qa = pipeline(model="impira/layoutlm-document-qa")
# # document_qa(
# #     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
# #     question="What is the invoice number?",top_k=3
# # )   


# # # import module
# # from pdf2image import convert_from_path, convert_from_bytes

# # with open('resume.pdf', 'rb') as file:
# #     new = file.read()

# # # Store Pdf with convert_from_path function
# # images = convert_from_path('resume.pdf')
# # images2 = convert_from_bytes(new)
 
# # images[0].save('page'+ str(1) +'.jpg', 'JPEG')
# # images[1].save('page'+ str(2) +'.jpg', 'JPEG')

# from transformers import pipeline

# oracle = pipeline(model="deepset/roberta-base-squad2")
# # small_oracle = pipeline(model="distilbert/distilbert-base-cased-distilled-squad")

# print(oracle(question="Where do I live?", context="I live in estonia and in India. ",
#        top_k=3))
# # {'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}

# full = 9
# # for full in range(1,10):
# total = (full)/3
# for index in range(full):
#        print(int(index//total))