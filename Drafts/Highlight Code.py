import fitz

pdfIn = fitz.open("resume.pdf")

for page in pdfIn:
    # print(page)
    texts = [ "5-person team"]
    text_instances = [page.search_for(text) for text in texts] 
    
    # coordinates of each word found in PDF-page
    # print(text_instances)  

    # iterate through each instance for highlighting
    for inst in text_instances:
        page.add_highlight_annot(inst)

# Saving the PDF Output
pdfIn.save("page-4_output.pdf")