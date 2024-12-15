import logging
import os
import sys
from pathlib import Path

from PBQA import DB, LLM  # run with python -m tests.rerank

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


tools = [
    "Search tool: Use the search tool to find information on a specific topic. The search tool is able to answer questions about current events, historical events, companies, products, and more. The search tool is able to search the web and provide a summary of the results. The search tool should not be used to answer personal questions.",
    "Weather tool: Use the weather tool to get the current weather conditions for a specific location. The weather tool is able to provide information on temperature, humidity, wind speed, and precipitation. The weather tool is able to provide a summary of the current weather conditions and any upcoming weather alerts or warnings.",
    "Email tool: Use the email tool to send and receive emails. The email tool is able to send emails to any recipient, including those who do not have an email address. The email tool is able to receive emails from any sender, including those who do not have an email address.",
    "Image generation tool: Use the image generation tool to generate images. The image generation tool uses a diffusion model to generate images based on a prompt. The prompt should be a description of what the image should look like.",
    "Optical character recognition (OCR) tool: Use the OCR tool to extract text from images. The OCR tool is able to extract text from images and convert them into editable text. The OCR tool is able to handle a wide range of image formats, including JPEG, PNG, GIF, and PDF.",
    "Text-to-speech tool: Use the text-to-speech tool to convert text into speech. The text-to-speech tool is able to convert text into speech and play it back as audio.",
    "Home Automation tool: Use the home automation tool to control devices in a home. The home automation tool is able to control lights, thermostats, security systems, and other devices in a home. The home automation tool can be used to both read the current state of a device and to change the state of a device.",
    "Spotify tool: Use the Spotify tool to play music. The Spotify tool is able to play music, control playback, and search for music.",
    "Document generation tool: Use the document generation tool to generate documents. Based on provided information and a prompt, the document generation tool can generate a document. The document may be in one of several formats, including DOCX, PDF, or HTML.",
    "Presentation tool: Use the presentation tool to create presentations. Using templates, the presentation tool can create a presentation with various slides and formatting options. The presentation tool creates a single presentation from provided information and a general prompt. The presentation tool can be used to create presentations in PDF or PPTX formats. Example usages include creating presentations for meetings, ",
    "Music generation tool: Use the music generation tool to generate music. The music generation tool can be used to create new music based on a prompt or existing music. The music generation tool can be used to create music in various formats, including MP3, WAV, and MIDI.",
    "Task management tool: Use the task management tool to manage tasks. The task management tool can be used to create, complete, and delete tasks.",
    "Note-taking tool: Use the note-taking tool to take notes. The note-taking tool can be used to create, read, and delete notes.",
    "Calendar tool: Use the calendar tool to manage events. The calendar tool can be used to create, read, and delete events. Example usages include scheduling meetings, finding upcoming personal events, setting reminders, and sending invitations.",
    "To-do list tool: Use the to-do list tool to manage tasks. The to-do list tool can be used to create, read, and delete tasks.",
    "Project management tool: Use the project management tool to manage projects. The project management tool can be used to create, read, and delete projects.",
    "Python REPL: Use the Python REPL to execute Python code. The Python REPL can be used to execute Python code and display the results. The Python REPL can be used to interact with a Python REPL in a web browser. Use the Python REPL to execute simple Python code, such as mathematical calculations or string operations, e.g., counting the number of words or letters in a string.",
]

db = DB(host="localhost", port=6333, reset=True)
llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="rerank",
    port=8080,
)

result = llm.rerank(
    "What's 10 + 20?",
    "rerank",
    tools,
    n=1,
)[0]

assert (
    result["index"] == 16
), f"Expected Python REPL to be the first result, got {result['index']}"

result = llm.rerank(
    "How many 'r's are there in 'strawberry'?",
    "rerank",
    tools,
    n=1,
)[0]

assert (
    result["index"] == 16
), f"Expected Python REPL to be the first result, got {result['index']}"


result = llm.rerank(
    "Turn off the lights",
    "rerank",
    tools,
    n=1,
)[0]

assert (
    result["index"] == 6
), f"Expected Home Automation tool to be the first result, got {result['index']}"

log.info("All tests passed")
