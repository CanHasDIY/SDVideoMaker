import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

import tkinter as tk
from tkinter import filedialog
import webbrowser
from threading import Thread
import sys

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
    """Get URL from text entry"""
    url = url_entry.get()
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()

        output_name = name_file.get() + '.mp4'

        # Load the conditioning image
        image = load_image(url)
        image = image.resize((1024, 576))
        
        app = Application(master=root)
        thread = Thread(target=app.update_output)
        thread.start()

        generator = torch.manual_seed(42)
        frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
        export_to_video(frames, output_name, fps=7)
        result = webbrowser.open(output_name)
    except ValueError as e:
        print("Invalid URL")

def browse_for_file():
    """Browse for file and get its path"""
    filename = filedialog.askopenfilename(title="Select file")
    if filename:
        url_entry.delete(0, tk.END)  # Clear text entry
        url_entry.insert(tk.END, filename)

root = tk.Tk()
root.title("Stable Diffusion Video Maker")

# Label and Entry for URL input
tk.Label(root, text="Enter a URL:").grid(row=0, column=0)
url_entry = tk.Entry(root, width=50)
url_entry.grid(row=0, column=1)

# Button to browse for file
tk.Label(root, text="Or browse for a file:").grid(row=1, column=0)
browse_button = tk.Button(root, text="Browse...", command=browse_for_file)
browse_button.grid(row=1, column=2)

# Name the output file
tk.Label(root, text="Output filename:").grid(row=3, column=0)
name_file = tk.Entry(root, width=50)
name_file.grid(row=3, column=1)


# Button to open URL
open_button = tk.Button(root, text="Create Video", command=open_url)
open_button.grid(row=4, column=1, columnspan=3)


root.mainloop()