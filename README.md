# ResuMate
An AI empowered resume analyser For fast, unbiased, effective and efficient hiring.
This was created as a submission for INNOV8, a hackathon by eightfold.ai

## Introduction
* The Resume Highlighting System has a Flask based frontend that allows users to upload a PDF resume and specify a focus area.
* The application then processes the resume, highlighting relevant texts based on the provided focus area. 
* Additionally, users can ask questions related to the resume, and the system will provide answers based on the resume content.

## Technologies Utilised
* Flask for frontend
* Roberta for question answering system
* MixedBread for sentence embedding
* Sentence Transformers for cos-sim based quantified retrieval

## How to run
1. install these python modules: `fitz, flask, pipeline, sentence_transformers`
2. `cd` into `./main/`
3. run `python app.py`
4. Access the frontend on your browser at `localhost:5000`


Find out more at https://docs.google.com/presentation/d/1lFMlYsVylh8GcHVli4ao7VQ_X9GZdZBo5H4jdBTlsHo/
