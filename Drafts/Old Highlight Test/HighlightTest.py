from PyPDF2 import PdfWriter, PdfReader
from PyPDF2Highlight import createHighlight, addHighlightToPage

pdfInput = PdfReader(open("resume.pdf", "rb"))
pdfOutput = PdfWriter()

page1 = pdfInput.pages[0]

# Text search
for page in pdfInput.pages:
    ### SEARCH
    text = "team"
    text_instances = page.search_for(text)

    ### HIGHLIGHT
    for inst in text_instances:
        # highlight = page.add_highlight_annot(inst)
        # highlight.update()
        print(inst)


highlight = createHighlight(100, 400, 400, 500, {
    "author": "",
    "contents": ""
})


addHighlightToPage(highlight, page1, pdfOutput)

pdfOutput.add_page(page1)

outputStream = open("output2.pdf", "wb")
pdfOutput.write(outputStream)