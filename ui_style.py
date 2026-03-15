import tkinter as tk
from tkinter import ttk

PALETTE = {
    "bg": "#72BF78",
    "panel": "#A0D683",
    "accent": "#FEFF9F",
    "text": "#1f2d2b",
    "muted": "#2f3e3b",
    "canvas_bg": "#f7f7f7",
}

FONTS = {
    "title": ("Arial", 16, "bold"),
    "subtitle": ("Arial", 12, "bold"),
    "body": ("Arial", 11),
    "mono": ("Consolas", 10),
}


def apply_theme(root):
    root.configure(bg=PALETTE["bg"])

    style = ttk.Style(root)
    style.theme_use("default")

    style.configure("TFrame", background=PALETTE["bg"])
    style.configure("Panel.TFrame", background=PALETTE["panel"])
    style.configure("TLabel", background=PALETTE["panel"], foreground=PALETTE["text"], font=FONTS["body"])
    style.configure("Title.TLabel", background=PALETTE["bg"], foreground=PALETTE["text"], font=FONTS["title"])
    style.configure("Subtitle.TLabel", background=PALETTE["panel"], foreground=PALETTE["text"], font=FONTS["subtitle"])
    style.configure("TButton", background=PALETTE["accent"], foreground=PALETTE["muted"])

    style.map(
        "TButton",
        background=[("active", PALETTE["accent"])],
        foreground=[("active", PALETTE["muted"])],
    )


def style_widget(widget, *, bg=None, fg=None, font=None, relief=None):
    if bg is not None:
        widget.configure(bg=bg)
    if fg is not None:
        widget.configure(fg=fg)
    if font is not None:
        widget.configure(font=font)
    if relief is not None:
        widget.configure(relief=relief)


def create_panel(parent):
    panel = tk.Frame(parent, bg=PALETTE["panel"])
    return panel
