import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ui_style import PALETTE, FONTS, apply_theme, create_panel, style_widget

from data_prep import (
	encode_non_numeric,
	fill_missing_values,
	drop_columns,
	load_data_csv,
	load_data_train_test,
	normalize_data_z_score,
	split_train_test,
	split_reshape_data,
)
from Network_tools import train_network, predict, compute_mse


class DataPrepGUI:
	def __init__(self, root):
		self.root = root
		self.root.title("Data Loader and Preprocessing")
		self.root.geometry("1100x780")
		self.root.minsize(900, 640)
		self.root.rowconfigure(0, weight=1)
		self.root.columnconfigure(0, weight=1)
		apply_theme(self.root)

		self.train_path = None
		self.test_path = None
		self.single_path = None
		self.train_df = None
		self.test_df = None
		self.col_names = []

		self.x_data = None
		self.y_data = None
		self.test_features = None
		self.y_test = None
		self.feature_names = []
		self.layer_dims = []
		self.trained_weights = None
		self.trained_biases = None
		self.loss_history = []
		self._train_anim_id = None
		self._stop_requested = False
		self._anim_layer_index = 0
		self._anim_direction = 1

		self.label_var = tk.StringVar(value="")
		self.mode_var = tk.StringVar(value="train_test")
		self.encode_var = tk.BooleanVar(value=False)
		self.fill_var = tk.BooleanVar(value=False)
		self.normalize_var = tk.BooleanVar(value=False)
		self.split_ratio_var = tk.StringVar(value="0.2")
		self.hidden_layers_var = tk.StringVar(value="16,16")
		self.input_dim_var = tk.StringVar(value="-")
		self.output_dim_var = tk.StringVar(value="-")
		self.epochs_var = tk.StringVar(value="200")
		self.learning_rate_var = tk.StringVar(value="0.01")
		self.train_status_var = tk.StringVar(value="Idle")

		self.preprocess_frame = tk.Frame(self.root, bg=PALETTE["bg"])
		self.architecture_frame = tk.Frame(self.root, bg=PALETTE["bg"])
		self.train_frame = tk.Frame(self.root, bg=PALETTE["bg"])

		self._build_preprocess_ui()
		self._build_architecture_ui()
		self._build_train_ui()
		self._show_frame(self.preprocess_frame)

	def _build_preprocess_ui(self):
		self.preprocess_frame.columnconfigure(1, weight=1)
		self.preprocess_frame.columnconfigure(3, weight=1)

		tk.Label(self.preprocess_frame, text="Load mode", bg=PALETTE["bg"], font=FONTS["subtitle"]).grid(
			row=0, column=0, sticky="w", padx=8, pady=6
		)
		self.mode_train_radio = tk.Radiobutton(
			self.preprocess_frame,
			text="Train + Test files",
			variable=self.mode_var,
			value="train_test",
			command=self._on_mode_change,
			bg=PALETTE["bg"],
			fg=PALETTE["text"],
		)
		self.mode_train_radio.grid(row=0, column=1, sticky="w", padx=8)
		self.mode_single_radio = tk.Radiobutton(
			self.preprocess_frame,
			text="Single file (split)",
			variable=self.mode_var,
			value="single",
			command=self._on_mode_change,
			bg=PALETTE["bg"],
			fg=PALETTE["text"],
		)
		self.mode_single_radio.grid(row=0, column=2, sticky="w", padx=8)

		tk.Label(self.preprocess_frame, text="Training CSV", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=1, column=0, sticky="w", padx=8, pady=6
		)
		self.train_path_label = tk.Label(
			self.preprocess_frame,
			text="No file selected",
			anchor="w",
			bg=PALETTE["bg"],
			fg=PALETTE["muted"],
		)
		self.train_path_label.grid(row=1, column=1, sticky="we", padx=8, pady=6)
		self.train_button = tk.Button(
			self.preprocess_frame, text="Browse", command=self._pick_train, bg=PALETTE["accent"]
		)
		self.train_button.grid(row=1, column=2, padx=8, pady=6)

		tk.Label(self.preprocess_frame, text="Testing CSV", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=2, column=0, sticky="w", padx=8, pady=6
		)
		self.test_path_label = tk.Label(
			self.preprocess_frame,
			text="No file selected",
			anchor="w",
			bg=PALETTE["bg"],
			fg=PALETTE["muted"],
		)
		self.test_path_label.grid(row=2, column=1, sticky="we", padx=8, pady=6)
		self.test_button = tk.Button(
			self.preprocess_frame, text="Browse", command=self._pick_test, bg=PALETTE["accent"]
		)
		self.test_button.grid(row=2, column=2, padx=8, pady=6)

		tk.Label(self.preprocess_frame, text="Single CSV", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=3, column=0, sticky="w", padx=8, pady=6
		)
		self.single_path_label = tk.Label(
			self.preprocess_frame,
			text="No file selected",
			anchor="w",
			bg=PALETTE["bg"],
			fg=PALETTE["muted"],
		)
		self.single_path_label.grid(row=3, column=1, sticky="we", padx=8, pady=6)
		self.single_button = tk.Button(
			self.preprocess_frame, text="Browse", command=self._pick_single, bg=PALETTE["accent"]
		)
		self.single_button.grid(row=3, column=2, padx=8, pady=6)

		tk.Label(self.preprocess_frame, text="Test split ratio", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=4, column=0, sticky="w", padx=8, pady=6
		)
		self.split_ratio_entry = tk.Entry(
			self.preprocess_frame, textvariable=self.split_ratio_var, width=8, bg=PALETTE["accent"]
		)
		self.split_ratio_entry.grid(row=4, column=1, sticky="w", padx=8, pady=6)

		tk.Label(self.preprocess_frame, text="Label column", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=5, column=0, sticky="w", padx=8, pady=6
		)
		self.label_menu = tk.OptionMenu(self.preprocess_frame, self.label_var, "")
		self.label_menu.config(width=24)
		style_widget(self.label_menu, bg=PALETTE["accent"], fg=PALETTE["muted"], relief=tk.FLAT)
		self.label_menu.grid(row=5, column=1, sticky="w", padx=8, pady=6)

		tk.Label(self.preprocess_frame, text="Drop columns", bg=PALETTE["bg"], font=FONTS["body"]).grid(
			row=6, column=0, sticky="nw", padx=8, pady=6
		)
		self.drop_listbox = tk.Listbox(
			self.preprocess_frame, selectmode="multiple", exportselection=False, height=6, bg=PALETTE["accent"]
		)
		self.drop_listbox.grid(row=6, column=1, sticky="we", padx=8, pady=6)
		self.drop_listbox.bind("<<ListboxSelect>>", lambda event: self._update_preview())
		self.drop_clear_button = tk.Button(
			self.preprocess_frame,
			text="Clear selection",
			command=self._clear_drop_selection,
			bg=PALETTE["accent"],
		)
		self.drop_clear_button.grid(row=6, column=2, sticky="w", padx=8)

		preview_label = tk.Label(
			self.preprocess_frame, text="Preview (train data)", bg=PALETTE["bg"], font=FONTS["subtitle"]
		)
		preview_label.grid(row=1, column=3, sticky="w", padx=8, pady=6)
		preview_frame = create_panel(self.preprocess_frame)
		preview_frame.grid(row=2, column=3, rowspan=8, sticky="nsew", padx=8, pady=6)
		preview_frame.columnconfigure(0, weight=1)
		preview_frame.rowconfigure(0, weight=1)
		self.preview_text = tk.Text(
			preview_frame, height=18, wrap="none", bg=PALETTE["accent"], font=FONTS["mono"]
		)
		self.preview_text.grid(row=0, column=0, sticky="nsew")
		preview_scroll_y = tk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
		preview_scroll_y.grid(row=0, column=1, sticky="ns")
		preview_scroll_x = tk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_text.xview)
		preview_scroll_x.grid(row=1, column=0, sticky="ew")
		self.preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)

		tk.Label(self.preprocess_frame, text="Preprocessing", bg=PALETTE["bg"], font=FONTS["subtitle"]).grid(
			row=7, column=0, sticky="w", padx=8, pady=6
		)
		tk.Checkbutton(
			self.preprocess_frame,
			text="Encode non-numeric columns",
			variable=self.encode_var,
			bg=PALETTE["bg"],
			fg=PALETTE["text"],
		).grid(row=7, column=1, sticky="w", padx=8)
		tk.Checkbutton(
			self.preprocess_frame,
			text="Fill missing values with median",
			variable=self.fill_var,
			bg=PALETTE["bg"],
			fg=PALETTE["text"],
		).grid(row=8, column=1, sticky="w", padx=8)
		tk.Checkbutton(
			self.preprocess_frame,
			text="Normalize data (Z-score)",
			variable=self.normalize_var,
			bg=PALETTE["bg"],
			fg=PALETTE["text"],
		).grid(row=9, column=1, sticky="w", padx=8)

		tk.Button(
			self.preprocess_frame, text="Process Data", command=self._process_data, bg=PALETTE["accent"]
		).grid(
			row=10, column=1, sticky="e", padx=8, pady=12
		)

		self._on_mode_change()

	def _build_architecture_ui(self):
		self.architecture_frame.columnconfigure(0, weight=1)
		self.architecture_frame.rowconfigure(2, weight=1)

		tk.Label(self.architecture_frame, text="Model architecture", bg=PALETTE["bg"], font=FONTS["title"]).grid(
			row=0, column=0, sticky="w", padx=8, pady=6
		)

		info_frame = create_panel(self.architecture_frame)
		info_frame.grid(row=1, column=0, sticky="we", padx=8, pady=4)
		info_frame.columnconfigure(7, weight=1)

		tk.Label(info_frame, text="Input size:", bg=PALETTE["panel"], font=FONTS["body"]).grid(row=0, column=0, sticky="w")
		tk.Label(info_frame, textvariable=self.input_dim_var, bg=PALETTE["panel"], font=FONTS["body"]).grid(
			row=0, column=1, sticky="w", padx=(4, 12)
		)
		tk.Label(info_frame, text="Output size:", bg=PALETTE["panel"], font=FONTS["body"]).grid(row=0, column=2, sticky="w")
		tk.Label(info_frame, textvariable=self.output_dim_var, bg=PALETTE["panel"], font=FONTS["body"]).grid(
			row=0, column=3, sticky="w", padx=(4, 12)
		)
		tk.Label(info_frame, text="Hidden layers:", bg=PALETTE["panel"], font=FONTS["body"]).grid(row=0, column=4, sticky="w")
		tk.Entry(info_frame, textvariable=self.hidden_layers_var, width=18, bg=PALETTE["accent"]).grid(
			row=0, column=5, sticky="w", padx=(4, 8)
		)
		tk.Button(info_frame, text="Update", command=self._update_architecture, bg=PALETTE["accent"]).grid(
			row=0, column=6, sticky="w"
		)

		self.arch_canvas = tk.Canvas(self.architecture_frame, height=320, bg=PALETTE["canvas_bg"])
		self.arch_canvas.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
		self.arch_canvas.bind("<Configure>", lambda event: self._draw_network())

		button_frame = create_panel(self.architecture_frame)
		button_frame.grid(row=3, column=0, sticky="e", padx=8, pady=10)
		tk.Button(button_frame, text="Back", command=self._show_preprocess_screen, bg=PALETTE["accent"]).pack(
			side="left", padx=(0, 8)
		)
		tk.Button(button_frame, text="Next: Train", command=self._next_train, bg=PALETTE["accent"]).pack(
			side="left"
		)

	def _build_train_ui(self):
		self.train_frame.columnconfigure(0, weight=1)
		self.train_frame.columnconfigure(1, weight=1)
		self.train_frame.rowconfigure(2, weight=1)

		tk.Label(self.train_frame, text="Train model", bg=PALETTE["bg"], font=FONTS["title"]).grid(
			row=0, column=0, sticky="w", padx=8, pady=6
		)

		settings_frame = create_panel(self.train_frame)
		settings_frame.grid(row=1, column=0, sticky="nw", padx=8, pady=4)

		tk.Label(settings_frame, text="Epochs", bg=PALETTE["panel"], font=FONTS["body"]).grid(
			row=0, column=0, sticky="w"
		)
		tk.Entry(settings_frame, textvariable=self.epochs_var, width=10).grid(
			row=0, column=1, sticky="w", padx=(4, 12)
		)
		tk.Label(settings_frame, text="Learning rate", bg=PALETTE["panel"], font=FONTS["body"]).grid(
			row=0, column=2, sticky="w"
		)
		tk.Entry(settings_frame, textvariable=self.learning_rate_var, width=10).grid(
			row=0, column=3, sticky="w", padx=(4, 12)
		)
		self.train_button = tk.Button(
			settings_frame, text="Start Training", command=self._start_training, bg=PALETTE["accent"]
		)
		self.train_button.grid(row=0, column=4, sticky="w")
		self.stop_button = tk.Button(settings_frame, text="Stop", command=self._stop_training, bg=PALETTE["accent"])
		self.stop_button.grid(row=0, column=5, sticky="w", padx=(6, 0))

		status_frame = create_panel(self.train_frame)
		status_frame.grid(row=2, column=0, sticky="nw", padx=8, pady=6)
		self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=240)
		self.progress.grid(row=0, column=0, sticky="w")
		tk.Label(status_frame, textvariable=self.train_status_var, bg=PALETTE["panel"], font=FONTS["body"]).grid(
			row=0, column=1, sticky="w", padx=(10, 0)
		)

		metrics_frame = tk.LabelFrame(
			self.train_frame, text="Test metrics", bg=PALETTE["panel"], fg=PALETTE["text"]
		)
		metrics_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)
		metrics_frame.columnconfigure(0, weight=1)
		metrics_frame.rowconfigure(0, weight=1)
		self.metrics_text = tk.Text(
			metrics_frame, height=6, wrap="word", bg=PALETTE["accent"], font=FONTS["mono"]
		)
		self.metrics_text.grid(row=0, column=0, sticky="nsew")
		self.metrics_text.insert("end", "Train to see results.")

		visual_frame = tk.LabelFrame(
			self.train_frame, text="Training visualizer", bg=PALETTE["panel"], fg=PALETTE["text"]
		)
		visual_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
		visual_frame.columnconfigure(0, weight=1)
		visual_frame.rowconfigure(1, weight=1)

		self.stats_text = tk.Text(
			visual_frame, height=4, wrap="word", bg=PALETTE["accent"], font=FONTS["mono"]
		)
		self.stats_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
		self.stats_text.insert("end", "Configure data and model to view stats.")

		self.train_canvas = tk.Canvas(visual_frame, height=220, bg=PALETTE["canvas_bg"])
		self.train_canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
		self.train_canvas.bind("<Configure>", lambda event: self._draw_train_visuals())

		sample_frame = tk.LabelFrame(
			self.train_frame, text="Random test sample", bg=PALETTE["panel"], fg=PALETTE["text"]
		)
		sample_frame.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=8, pady=6)
		sample_frame.columnconfigure(0, weight=1)
		sample_frame.rowconfigure(1, weight=1)
		self.sample_button = tk.Button(
			sample_frame, text="Show Random Sample", command=self._show_random_sample, bg=PALETTE["accent"]
		)
		self.sample_button.grid(row=0, column=0, sticky="w", padx=6, pady=6)
		self.sample_text = tk.Text(sample_frame, wrap="word", bg=PALETTE["accent"], font=FONTS["mono"])
		self.sample_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
		self.sample_text.insert("end", "Train the model to view predictions.")

		back_button = tk.Button(
			self.train_frame, text="Back", command=self._show_architecture_screen, bg=PALETTE["accent"]
		)
		back_button.grid(row=4, column=0, sticky="w", padx=8, pady=10)

	def _show_frame(self, frame):
		for child in self.root.winfo_children():
			child.grid_forget()
		frame.grid(row=0, column=0, sticky="nsew")

	def _show_architecture_screen(self):
		self.root.title("Model Architecture")
		self._update_architecture_defaults()
		self._show_frame(self.architecture_frame)

	def _show_train_screen(self):
		self.root.title("Train Model")
		self._update_train_stats()
		self._draw_train_visuals()
		self._show_frame(self.train_frame)

	def _show_preprocess_screen(self):
		self.root.title("Data Loader and Preprocessing")
		self._show_frame(self.preprocess_frame)

	def _pick_train(self):
		path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
		if path:
			self.train_path = path
			self._set_path_label(self.train_path_label, path)
			self._load_data_if_ready()

	def _pick_test(self):
		path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
		if path:
			self.test_path = path
			self._set_path_label(self.test_path_label, path)
			self._load_data_if_ready()

	def _pick_single(self):
		path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
		if path:
			self.single_path = path
			self._set_path_label(self.single_path_label, path)
			self._load_data_if_ready()

	def _set_path_label(self, label, path, max_len=48):
		display = path
		if len(path) > max_len:
			keep = max_len - 3
			head = keep // 2
			tail = keep - head
			display = f"{path[:head]}...{path[-tail:]}"
		label.config(text=display)

	def _on_mode_change(self):
		mode = self.mode_var.get()
		train_state = "normal" if mode == "train_test" else "disabled"
		test_state = "normal" if mode == "train_test" else "disabled"
		single_state = "normal" if mode == "single" else "disabled"
		split_state = "normal" if mode == "single" else "disabled"

		self.train_button.config(state=train_state)
		self.test_button.config(state=test_state)
		self.single_button.config(state=single_state)
		self.split_ratio_entry.config(state=split_state)

		self.train_df = None
		self.test_df = None
		self.col_names = []
		self._refresh_label_menu()
		self._refresh_drop_list()
		self._update_preview()

	def _load_data_if_ready(self):
		mode = self.mode_var.get()
		if mode == "train_test":
			if not self.train_path or not self.test_path:
				return
			try:
				self.train_df, self.test_df, _ = load_data_train_test(
					self.train_path, self.test_path
				)
				self.col_names = list(self.train_df.columns)
				self._refresh_label_menu()
				self._refresh_drop_list()
				self._update_preview()
			except Exception as exc:
				messagebox.showerror("Load Error", f"Failed to load CSV files: {exc}")
		else:
			if not self.single_path:
				return
			try:
				data_df = load_data_csv(self.single_path)
				test_ratio = float(self.split_ratio_var.get())
				self.train_df, self.test_df = split_train_test(data_df, test_ratio=test_ratio)
				self.col_names = list(self.train_df.columns)
				self._refresh_label_menu()
				self._refresh_drop_list()
				self._update_preview()
			except Exception as exc:
				messagebox.showerror("Load Error", f"Failed to load CSV file: {exc}")

	def _refresh_label_menu(self):
		menu = self.label_menu["menu"]
		menu.delete(0, "end")
		for name in self.col_names:
			menu.add_command(label=name, command=lambda value=name: self.label_var.set(value))
		if len(self.col_names) > 0:
			self.label_var.set(self.col_names[0])
		else:
			self.label_var.set("")

	def _refresh_drop_list(self):
		self.drop_listbox.delete(0, "end")
		for name in self.col_names:
			self.drop_listbox.insert("end", name)
		self._update_preview()

	def _clear_drop_selection(self):
		self.drop_listbox.selection_clear(0, "end")
		self._update_preview()

	def _update_preview(self):
		if self.train_df is None:
			self.preview_text.delete("1.0", "end")
			self.preview_text.insert("end", "Load data to preview.")
			return

		drop_indices = self.drop_listbox.curselection()
		drop_cols = [self.drop_listbox.get(i) for i in drop_indices]
		preview_df = drop_columns(self.train_df, drop_cols)
		preview_str = preview_df.head().to_string(index=False)
		self.preview_text.delete("1.0", "end")
		self.preview_text.insert("end", preview_str)

	def _process_data(self):
		if self.train_df is None or self.test_df is None:
			messagebox.showwarning("Missing Files", "Please load data before processing.")
			return

		label = self.label_var.get().strip()
		if not label or label not in self.train_df.columns:
			messagebox.showerror("Invalid Label", "Selected label column was not found in training data.")
			raise ValueError("Selected label column does not exist in training data.")

		labels = [label]
		train_df = self.train_df.copy()
		test_df = self.test_df.copy()

		drop_indices = self.drop_listbox.curselection()
		drop_cols = [self.drop_listbox.get(i) for i in drop_indices]
		if label in drop_cols:
			messagebox.showwarning(
				"Drop Columns",
				"Label column cannot be dropped. It will be kept for training.",
			)
			drop_cols = [col for col in drop_cols if col != label]

		train_df = drop_columns(train_df, drop_cols)
		test_df = drop_columns(test_df, drop_cols)
		self.feature_names = [col for col in train_df.columns if col != label]

		if self.encode_var.get():
			train_df = encode_non_numeric(train_df)
			test_df = encode_non_numeric(test_df)

		if self.fill_var.get():
			train_df = fill_missing_values(train_df)
			test_df = fill_missing_values(test_df)

		if self.normalize_var.get():
			train_df, mean, std = normalize_data_z_score(train_df, labels=labels)
			test_labels = test_df[labels].to_numpy() if label in test_df.columns else None
			test_features = test_df.drop(columns=labels, errors="ignore")
			test_df = test_features.copy()
			test_df = (test_df - mean) / std
			self.y_test = test_labels.T if test_labels is not None else None
		else:
			self.y_test = test_df[labels].to_numpy().T if label in test_df.columns else None
			test_df = test_df.drop(columns=labels, errors="ignore")

		self.x_data, self.y_data = split_reshape_data(train_df, labels)
		self.test_features = test_df.to_numpy().T

		messagebox.showinfo("Success", "Data processed and stored in memory.")
		self._show_architecture_screen()

	def _update_architecture_defaults(self):
		if self.x_data is None or self.y_data is None:
			return
		input_dim = int(self.x_data.shape[0])
		output_dim = int(self.y_data.shape[0])
		self.input_dim_var.set(str(input_dim))
		self.output_dim_var.set(str(output_dim))
		self._update_architecture()

	def _update_architecture(self):
		if self.x_data is None or self.y_data is None:
			return

		input_dim = int(self.x_data.shape[0])
		output_dim = int(self.y_data.shape[0])
		self.input_dim_var.set(str(input_dim))
		self.output_dim_var.set(str(output_dim))

		text = self.hidden_layers_var.get().strip()
		if text:
			try:
				hidden_layers = [int(item) for item in text.split(",") if item.strip()]
				if any(size <= 0 for size in hidden_layers):
					raise ValueError("Hidden layer sizes must be positive.")
			except ValueError:
				messagebox.showerror(
					"Invalid Hidden Layers",
					"Enter comma-separated positive integers (e.g. 32,16).",
				)
				return
		else:
			hidden_layers = []

		self.layer_dims = [input_dim] + hidden_layers + [output_dim]
		self._draw_network()

	def _draw_network(self):
		if not self.layer_dims:
			return

		canvas = self.arch_canvas
		canvas.delete("all")
		width = canvas.winfo_width()
		height = canvas.winfo_height()
		if width <= 2 or height <= 2:
			return

		layer_count = len(self.layer_dims)
		max_display = 8
		margin_x = 30
		margin_y = 24
		x_spacing = (width - 2 * margin_x) / max(layer_count - 1, 1)
		node_radius = 8

		layer_positions = []
		for layer_index, layer_size in enumerate(self.layer_dims):
			display_size = min(layer_size, max_display)
			y_spacing = (height - 2 * margin_y) / max(display_size - 1, 1)
			x = margin_x + layer_index * x_spacing
			positions = []
			for node_index in range(display_size):
				y = margin_y + node_index * y_spacing
				positions.append((x, y))
			layer_positions.append(positions)

		for layer_index in range(layer_count - 1):
			for start in layer_positions[layer_index]:
				for end in layer_positions[layer_index + 1]:
					canvas.create_line(start[0], start[1], end[0], end[1], fill="#7f8c8d")

		for layer_index, positions in enumerate(layer_positions):
			for (x, y) in positions:
				canvas.create_oval(
					x - node_radius,
					y - node_radius,
					x + node_radius,
					y + node_radius,
					fill="#2c3e50",
					outline="",
				)
			layer_size = self.layer_dims[layer_index]
			canvas.create_text(
				positions[0][0],
				margin_y - 10,
				text=str(layer_size),
				fill="#2c3e50",
				font=("Arial", 9, "bold"),
			)
			if layer_size > max_display:
				canvas.create_text(
					positions[0][0],
					height - margin_y + 10,
					text=f"+{layer_size - max_display} more",
					fill="#7f8c8d",
					font=("Arial", 8),
				)

	def _next_train(self):
		self._update_architecture()
		if not self.layer_dims:
			messagebox.showwarning("Missing Architecture", "Please configure the model architecture first.")
			return
		self._show_train_screen()

	def _update_train_stats(self):
		self.stats_text.delete("1.0", "end")
		if self.x_data is None or self.y_data is None:
			self.stats_text.insert("end", "Load data to view training stats.")
			return

		samples = self.x_data.shape[1]
		features = self.x_data.shape[0]
		outputs = self.y_data.shape[0]
		layers = ", ".join(str(dim) for dim in self.layer_dims) if self.layer_dims else "-"
		stats_lines = [
			f"Samples: {samples}",
			f"Features: {features}",
			f"Outputs: {outputs}",
			f"Architecture: {layers}",
		]
		self.stats_text.insert("end", "\n".join(stats_lines))

	def _start_training(self):
		if self.x_data is None or self.y_data is None or not self.layer_dims:
			messagebox.showwarning("Missing Data", "Please preprocess data and configure the model.")
			return

		try:
			epochs = int(self.epochs_var.get())
			learning_rate = float(self.learning_rate_var.get())
			if epochs <= 0 or learning_rate <= 0:
				raise ValueError
		except ValueError:
			messagebox.showerror("Invalid Training", "Enter positive values for epochs and learning rate.")
			return

		self.train_button.config(state="disabled")
		self.sample_button.config(state="disabled")
		self.stop_button.config(state="normal")
		self.train_status_var.set("Training...")
		self.progress.start(10)
		self.loss_history = []
		self._stop_requested = False

		thread = threading.Thread(
			target=self._run_training,
			args=(epochs, learning_rate),
			daemon=True,
		)
		thread.start()
		self._start_train_animation()

	def _run_training(self, epochs, learning_rate):
		def progress_callback(epoch, total_epochs, loss):
			self.root.after(0, lambda: self._on_train_progress(epoch, total_epochs, loss))

		def stop_flag():
			return self._stop_requested

		try:
			weights, biases, loss_history = train_network(
				self.x_data,
				self.y_data,
				self.layer_dims,
				epochs=epochs,
				learning_rate=learning_rate,
				progress_callback=progress_callback,
				stop_flag=stop_flag,
			)

			self.trained_weights = weights
			self.trained_biases = biases

			self.root.after(0, lambda: self._finish_training(loss_history))
		except Exception as exc:
			self.root.after(0, lambda: self._handle_training_error(exc))

	def _handle_training_error(self, exc):
		self.progress.stop()
		self.train_status_var.set("Training failed")
		self.train_button.config(state="normal")
		self.sample_button.config(state="normal")
		self.stop_button.config(state="disabled")
		self._stop_train_animation()
		messagebox.showerror("Training Error", f"Training failed: {exc}")

	def _on_train_progress(self, epoch, total_epochs, loss):
		self.loss_history.append(loss)
		self.train_status_var.set(f"Training... {epoch}/{total_epochs} | Loss: {loss:.6f}")
		self._draw_train_visuals(animated=True, highlight_index=self._anim_layer_index)

	def _finish_training(self, loss_history):
		self.progress.stop()
		status_text = "Training stopped" if self._stop_requested else "Training complete"
		self.train_status_var.set(status_text)
		self.train_button.config(state="normal")
		self.sample_button.config(state="normal")
		self.stop_button.config(state="disabled")
		self.loss_history = loss_history
		self._stop_train_animation()
		self._draw_train_visuals()

		metrics_lines = []
		metrics_lines.append(f"Final train loss (MSE): {loss_history[-1]:.6f}")

		if self.test_features is not None and self.trained_weights is not None:
			y_pred = predict(self.test_features, self.trained_weights, self.trained_biases)
			if self.y_test is not None:
				test_loss = compute_mse(self.y_test, y_pred)
				metrics_lines.append(f"Test loss (MSE): {test_loss:.6f}")
				if self.y_test.shape[0] == 1:
					unique_vals = set(self.y_test.flatten().tolist())
					if unique_vals.issubset({0.0, 1.0}):
						pred_labels = (y_pred >= 0.5).astype(int)
						accuracy = np.mean(pred_labels == self.y_test)
						metrics_lines.append(f"Test accuracy: {accuracy * 100:.2f}%")
			else:
				metrics_lines.append("Test labels not available; skipped test metrics.")

		self.metrics_text.delete("1.0", "end")
		self.metrics_text.insert("end", "\n".join(metrics_lines))

	def _start_train_animation(self):
		self._stop_train_animation()
		self._anim_layer_index = 0
		self._anim_direction = 1
		self._train_anim_id = self.root.after(120, self._animate_train_visuals)

	def _stop_train_animation(self):
		if self._train_anim_id is not None:
			self.root.after_cancel(self._train_anim_id)
			self._train_anim_id = None

	def _animate_train_visuals(self):
		layer_count = len(self.layer_dims)
		if layer_count > 0:
			self._anim_layer_index += self._anim_direction
			if self._anim_layer_index >= layer_count - 1:
				self._anim_direction = -1
				self._anim_layer_index = layer_count - 1
			elif self._anim_layer_index <= 0:
				self._anim_direction = 1
				self._anim_layer_index = 0
		self._draw_train_visuals(animated=True, highlight_index=self._anim_layer_index)
		self._train_anim_id = self.root.after(220, self._animate_train_visuals)

	def _draw_train_visuals(self, animated=False, highlight_index=None):
		canvas = self.train_canvas
		canvas.delete("all")
		width = canvas.winfo_width()
		height = canvas.winfo_height()
		if width <= 2 or height <= 2:
			return

		margin = 16
		layer_dims = self.layer_dims
		if not layer_dims:
			canvas.create_text(
				width / 2,
				height / 2,
				text="Configure a model to visualize training.",
				fill="#7f8c8d",
			)
			return

		layer_count = len(layer_dims)
		layer_x = [
			margin + idx * (width - 2 * margin) / max(layer_count - 1, 1)
			for idx in range(layer_count)
		]

		net_height = height * 0.55
		center_y = net_height * 0.45
		node_radius = 8
		max_display = 6

		if animated and highlight_index is None:
			highlight_index = 0

		for idx, dim in enumerate(layer_dims):
			display_nodes = min(dim, max_display)
			spacing = 20
			start_y = center_y - (display_nodes - 1) * spacing / 2
			glow = "#ffd166" if idx == highlight_index else "#dfe6e9"
			for node in range(display_nodes):
				y = start_y + node * spacing
				canvas.create_oval(
					layer_x[idx] - node_radius,
					y - node_radius,
					layer_x[idx] + node_radius,
					y + node_radius,
					fill=glow,
					outline="#7f8c8d",
				)
			if dim > max_display:
				canvas.create_text(
					layer_x[idx],
					start_y + display_nodes * spacing,
					text=f"+{dim - max_display}",
					fill="#7f8c8d",
					font=("Arial", 8),
				)
			canvas.create_text(
				layer_x[idx],
				margin,
				text=str(dim),
				fill="#2c3e50",
				font=("Arial", 9, "bold"),
			)

		for idx in range(layer_count - 1):
			canvas.create_line(
				layer_x[idx],
				center_y,
				layer_x[idx + 1],
				center_y,
				fill="#b2bec3",
				width=3,
			)

		canvas.create_text(
			width / 2,
			net_height - margin,
			text="Training activity",
			fill="#7f8c8d",
			font=("Arial", 9),
		)

		# Loss plot beneath the network diagram
		loss_top = net_height + margin
		loss_bottom = height - margin
		canvas.create_rectangle(
			margin,
			loss_top,
			width - margin,
			loss_bottom,
			outline="#dfe6e9",
			fill="",
		)

		if len(self.loss_history) >= 2:
			losses = self.loss_history[-200:]  # keep it light
			min_loss = min(losses)
			max_loss = max(losses)
			span = max(max_loss - min_loss, 1e-9)
			x_step = (width - 2 * margin) / max(len(losses) - 1, 1)
			points = []
			for i, loss in enumerate(losses):
				norm = (loss - min_loss) / span
				x = margin + i * x_step
				y = loss_bottom - norm * (loss_bottom - loss_top)
				points.append((x, y))
			for i in range(len(points) - 1):
				canvas.create_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], fill="#2d3436", width=2)
			canvas.create_text(
				margin + 4,
				loss_top + 10,
				text=f"Loss: {losses[-1]:.4f}",
				anchor="w",
				fill="#2c3e50",
				font=("Arial", 9, "bold"),
			)
		else:
			canvas.create_text(
				width / 2,
				(loss_top + loss_bottom) / 2,
				text="Loss curve appears during training",
				fill="#7f8c8d",
				font=("Arial", 9),
			)

	def _stop_training(self):
		self._stop_requested = True
		self.train_status_var.set("Stopping...")

	def _show_random_sample(self):
		if self.test_features is None or self.trained_weights is None:
			messagebox.showwarning("No Model", "Train the model before previewing samples.")
			return

		index = int(np.random.randint(0, self.test_features.shape[1]))
		sample = self.test_features[:, index:index + 1]
		prediction = predict(sample, self.trained_weights, self.trained_biases)

		lines = []
		lines.append(f"Sample index: {index}")
		if self.feature_names:
			for name, value in zip(self.feature_names, sample.flatten().tolist()):
				lines.append(f"{name}: {value:.4f}")
		else:
			lines.append("Features:")
			lines.extend([f"{value:.4f}" for value in sample.flatten().tolist()])

		if self.y_test is not None:
			actual = self.y_test[:, index]
			lines.append(f"Actual label: {actual.flatten().tolist()}")
		lines.append(f"Prediction: {prediction.flatten().tolist()}")

		self.sample_text.delete("1.0", "end")
		self.sample_text.insert("end", "\n".join(lines))


if __name__ == "__main__":
	root = tk.Tk()
	app = DataPrepGUI(root)
	root.mainloop()
