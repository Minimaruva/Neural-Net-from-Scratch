# This is gui from one of my other projects
# I want to use the color pallate and style in our project
# but export styles as css for better separation of concertns:


import tkinter as tk
# Import Scrollbar
from tkinter import filedialog, simpledialog, messagebox, Scrollbar
import os
from PIL import Image as PILImage
from moviepy.editor import VideoFileClip
import pandas as pd
import random
from to_video import img_to_img, img_to_video, video_to_video
from text_generator import create_text_image, random_quote as get_random_quote

# Default output path
output_path = "./output"
file_path = None  

def random_quote():
    global quote_entry
    try:
        quote = get_random_quote("./assets/Quotes.csv")
    except Exception as e:
        quote = "Quotes not found"
        print(f"Error loading quotes: {e}")

    quote_entry.delete("1.0", tk.END)
    quote_entry.insert("1.0", quote)
    
def import_file():
    global file_path 
    file_path = filedialog.askopenfilename(title="Select a file")
    
    # Show selected filename in UI
    if file_path:
        file_label.config(text=f"Selected: {os.path.basename(file_path)}")

def output_folder():
    global output_path
    selected_path = filedialog.askdirectory(title="Select the output path")
    if selected_path:  # Only update if user selected something
        output_path = selected_path
        output_path_label.config(text=f"Output: {output_path}")

def edit():
    global file_path
    global output_path
    global selected_value
    
    # Check if file was selected
    if not file_path:
        messagebox.showerror("Error", "Please select an input file first")
        return
    
    choice = selected_value.get()
    quote_text = quote_entry.get("1.0", tk.END).strip()
    
    # Validate duration
    try:
        duration = duration_entry.get()
        if duration == "":
            duration = 10
        else:
            duration = int(duration)
            if duration <= 0:
                raise ValueError("Duration must be positive")
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid duration: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Determine target width for quote image to match background or video frame
        if choice in ["img-to-img", "img-to-video"]:
            bg_img = PILImage.open(file_path)
            target_width = bg_img.width
        elif choice == "video-to-video":
            clip = VideoFileClip(file_path)
            target_width = clip.w
        else:
            target_width = None
        # Create quote image with dynamic width
        quote_path = create_text_image(text=quote_text, target_width=target_width)
        
        # Process based on selected operation
        if choice == "img-to-img":
            img_to_img(quote_path, img_path=file_path, output_path=output_path)
            messagebox.showinfo("Success", f"Image saved to {output_path}")
            root.quit()
        elif choice == "img-to-video":
            img_to_video(quote_path, duration=duration, img_path=file_path, output_path=output_path)
            messagebox.showinfo("Success", f"Video saved to {output_path}")
            root.quit()
        elif choice == "video-to-video":
            video_to_video(quote_path, duration=duration, video_path=file_path, output_path=output_path)
            messagebox.showinfo("Success", f"Video saved to {output_path}")
            root.quit()
        else:
            messagebox.showinfo("Error", "Please choose the editing mode (e.g. img-to-img)")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}\n\nMake sure you have selected a file and chosen the correct action.")
        print(f"Error details: {e}")

# Set up GUI
root = tk.Tk()
root.title("EasyReel by Vlad")
root.geometry("400x500")
root.configure(bg='#72BF78')

app_label = tk.Label(root, text="EasyReel\nMaking quoted content has never been easier!", font=("Arial", 14), bg='#72BF78')
app_label.pack()

# Quote frame
quote_frame = tk.Frame(root, bg="#A0D683")
quote_label = tk.Label(quote_frame, text="Enter your quote here", bg="#A0D683")

# Create a frame to hold the text widget and scrollbar
text_scroll_frame = tk.Frame(quote_frame, bg="#A0D683")

# Create Scrollbar with colors
scrollbar = Scrollbar(text_scroll_frame, bg="#FEFF9F", troughcolor="#A0D683")
# Pack scrollbar to the right
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure Text widget with reduced width and scrollbar
quote_entry = tk.Text(text_scroll_frame, width=43, height=3, bg="#FEFF9F", wrap=tk.WORD, yscrollcommand=scrollbar.set)
# Pack text entry to the left, expanding to fill remaining space
quote_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Configure Scrollbar command
scrollbar.config(command=quote_entry.yview)

random_quote_button = tk.Button(quote_frame, text="Random quote", command=random_quote, bg="#FEFF9F", relief=tk.FLAT)

quote_label.pack(pady=5)
# Pack the frame containing text and scrollbar instead of just the text widget
text_scroll_frame.pack(padx=5)
random_quote_button.pack(pady=10)
quote_frame.pack()

# File uploader frame
file_frame = tk.Frame(root, bg="#A0D683")
file_label = tk.Label(file_frame, text="Choose your file (photo or image)", bg="#A0D683")
file_button = tk.Button(file_frame, text="Import file", command=import_file, bg="#FEFF9F", relief=tk.FLAT)

duration_label = tk.Label(file_frame, text="Enter duration of video in seconds (optional)", bg="#A0D683")
duration_entry = tk.Entry(file_frame, width=10, bg="#FEFF9F")

# Frame inside of File frame for radio buttons
radio_frame = tk.Frame(file_frame, bg="#A0D683")
selected_value = tk.StringVar()
rb1 = tk.Radiobutton(radio_frame, text="img-to-img", variable=selected_value, value="img-to-img", bg="#FEFF9F")
rb2 = tk.Radiobutton(radio_frame, text="img-to-video", variable=selected_value, value="img-to-video", bg="#FEFF9F")
rb3 = tk.Radiobutton(radio_frame, text="video-to-video", variable=selected_value, value="video-to-video", bg="#FEFF9F")


file_label.pack(padx=94)
file_button.pack(pady=10)
duration_label.pack(padx=10)
duration_entry.pack(padx=10)

rb1.grid(row=0, column=0, padx=10)
rb2.grid(row=0, column=1, padx=10)
rb3.grid(row=0, column=2, padx=10)
radio_frame.pack(padx=5, pady=10)

file_frame.pack(pady=10)

# Submit Frame
submit_frame = tk.Frame(root, bg="#A0D683")
output_path_label = tk.Label(submit_frame, text="Choose output folder (optional)", bg="#A0D683")
output_path_button = tk.Button(submit_frame, text="Choose output folder", command=output_folder, bg="#FEFF9F", relief=tk.FLAT)

submit_button = tk.Button(submit_frame, text="Submit for edit", command=edit, bg="#FEFF9F", relief=tk.GROOVE)

output_path_label.pack(padx=99)
output_path_button.pack(pady=10)

submit_button.pack(padx=10, pady=10)

submit_frame.pack(pady=10)

root.mainloop()