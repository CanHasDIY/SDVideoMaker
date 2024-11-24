import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

import tkinter as tk
from tkinter import filedialog
import webbrowser
from threading import Thread
import sys

# Create a frame in the GUI window to display console output
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
#        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Create a new Text widget to display console output
        self.output_text = tk.Text(width=60, height=10)
#        self.output_text.pack(side="top")

    def update_output(self):
        while True:
            line = sys.stdout.readline()
            if not line:
                break
            self.output_text.insert(tk.END, line)

def open_url():
    # Get input from text entry
    url = url_entry.get()
    try:
        # Load and configure the Stable Diffusion model
        pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.float16, 
        variant="fp16"
        ).to("cuda")
        pipe.enable_model_cpu_offload()

        # Get the output filename and append '.mp4' extension
        output_name = name_file.get() + '.mp4'

        # Load the conditioning image
        image = load_image(url)
        image = image.resize((1024, 576))
        
        # Display the console output in the GUI window
        app = Application(master=root)
        thread = Thread(target=app.update_output)
        thread.start()

        # Generate the AI video frames
        generator = torch.manual_seed(42)
        frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
        
        # Create the output file and display it in a web browser
        export_to_video(frames, output_name, fps=7)
        result = webbrowser.open(output_name)
    except ValueError as e:
        print("Invalid URL or filename")

def browse_for_file():
    # Browse for file and get its path
    filename = filedialog.askopenfilename(title="Select file")
    if filename:
        url_entry.delete(0, tk.END)  # Clear text entry
        url_entry.insert(tk.END, filename)

# Draw the GUI window
root = tk.Tk()
root.title("Stable Diffusion Video Maker")

# Label and Entry for input
tk.Label(root, text="Enter a URL:").grid(row=0, column=0)
url_entry = tk.Entry(root, width=20)
url_entry.grid(row=0, column=1)

# Button to browse for file
tk.Label(root, text="Or browse for a file:").grid(row=2, column=0)
browse_button = tk.Button(root, text="Browse...", command=browse_for_file)
browse_button.grid(row=2, column=1)

# Name the output file
tk.Label(root, text="Output filename:").grid(row=4, column=0)
name_file = tk.Entry(root, width=20)
name_file.grid(row=4, column=1)
tk.Label(root, text=".mp4").grid(row=4, column=2)

# Button to open URL
open_button = tk.Button(root, text="Create Video", command=open_url)
open_button.grid(row=5, column=1, columnspan=2)

col_count, row_count = root.grid_size()

for col in range(col_count):
    root.grid_columnconfigure(col, minsize=10)

for row in range(row_count):
    root.grid_rowconfigure(row, minsize=10)


root.mainloop()