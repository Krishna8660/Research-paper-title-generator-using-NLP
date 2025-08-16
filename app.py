import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import gradio as gr
import re
import matplotlib.pyplot as plt

# Load and clean dataset
file_path = r"C:\Users\DELL\Downloads\archive (4)\arXiv_scientific dataset.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
df = df.rename(columns={'summary': 'abstract', 'title': 'original_title'})
df = df[['abstract', 'original_title']].dropna().drop_duplicates()
df = df.head(500).reset_index(drop=True)
print(f"Loaded {len(df)} samples.")

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(r"D:\nlp project\gpt2_lora_title_model")
tokenizer = GPT2Tokenizer.from_pretrained(r"D:\nlp project\gpt2_lora_title_model")
model.to(device)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- CACHES ---
title_cache = {}           # cache for generate_title_lora by abstract text
dataset_cache = {}         # cache for dataset indexed results
batch_cache = {}           # cache for batch results by index range

def generate_title_lora(abstract):
    # Check cache first
    if abstract in title_cache:
        return title_cache[abstract]

    model.eval()
    prompt = (
        "Given the following abstract, generate a concise academic paper title "
        "(no more than 12 words):\n\n"
        f"{abstract}\n\nTitle:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=2,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Title:\s*(.*)", decoded, re.IGNORECASE)
    title = match.group(1).strip() if match else decoded.strip()
    title = re.sub(r'[^\w\s-]', '', title).strip()
    title_tokens = title.split()
    if len(title_tokens) > 15:
        title = ' '.join(title_tokens[:15])

    # Store in cache
    title_cache[abstract] = title
    return title

def generate_from_dataset_with_abstract(index):
    # Check cache
    if index in dataset_cache:
        return dataset_cache[index]

    row = df.iloc[int(index)]
    abstract = row['abstract']
    original_title = row['original_title']
    generated_title = generate_title_lora(abstract)
    embeddings = sbert_model.encode([original_title, generated_title], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(['SBERT Similarity'], [similarity], color='dodgerblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Similarity Score")
    ax.set_title("SBERT Similarity")
    ax.text(0, similarity + 0.02, f"{similarity:.3f}", ha='center', fontsize=10)
    plt.tight_layout()

    result = (abstract, original_title, generated_title, similarity, fig)
    dataset_cache[index] = result
    return result

def generate_batch_titles(index_range):
    # Check cache
    if index_range in batch_cache:
        return batch_cache[index_range]

    try:
        start, end = [int(x.strip()) for x in index_range.split("-")]
        if start > end or start < 0 or end >= len(df):
            return "Invalid index range. Use format like 0-5 within dataset size.", "", None, None
    except:
        return "Invalid input format. Use 'start-end' like 0-5", "", None, None

    results = []
    similarities = []
    for idx in range(start, end + 1):
        row = df.iloc[idx]
        gen_title = generate_title_lora(row['abstract'])
        emb = sbert_model.encode([row['original_title'], gen_title], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb[0], emb[1]).item()
        results.append((idx, row['abstract'], row['original_title'], gen_title, sim))
        similarities.append(sim)

    fig, ax = plt.subplots(figsize=(8,4))
    indices = [str(r[0]) for r in results]
    sims = [r[4] for r in results]
    ax.bar(indices, sims, color='mediumseagreen')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset Index")
    ax.set_ylabel("SBERT Similarity")
    ax.set_title(f"SBERT Similarity Scores for indices {start}-{end}")
    for i, v in enumerate(sims):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
    plt.tight_layout()

    output_text = ""
    for idx, abstract, orig, gen, sim in results:
        output_text += (
            f"Index {idx}\n"
            f"Abstract: {abstract}\n"
            f"Original Title: {orig}\n"
            f"Generated Title: {gen}\n"
            f"Similarity: {sim:.3f}\n\n"
        )

    result = (output_text, "", None, fig)
    batch_cache[index_range] = result
    return result

def generate_from_abstract(abstract):
    gen_title = generate_title_lora(abstract)
    return gen_title, "", None, None

def clear_dataset():
    return "", "", "", None, None

def clear_batch():
    return "", "", None, None

def clear_abstract():
    return "",

with gr.Blocks(css="body {background-color: white;}") as iface:
    gr.Markdown(
        "<h1 style='text-align: center; color: white; font-family: Verdana;'>"
        "Research Paper Title Generator</h1>",
        elem_id="main-title"
    )

    with gr.Tab("From Dataset Sample"):
        idx_input = gr.Dropdown(choices=[str(i) for i in df.index], label="Select Dataset Sample Index")
        with gr.Row():
            btn_submit_ds = gr.Button("Submit", variant="primary")
            btn_clear_ds = gr.Button("Clear")

        abstract_ds = gr.Textbox(label="Abstract", lines=5, interactive=False)
        orig_title_ds = gr.Textbox(label="Original Title", interactive=False)
        gen_title_ds = gr.Textbox(label="Generated Title", interactive=False, elem_id="generated-title-box")
        sbert_score_ds = gr.Number(label="SBERT Similarity Score", precision=3, interactive=False)
        sbert_plot_ds = gr.Plot(label="SBERT Similarity Graph")

        btn_submit_ds.click(
            fn=generate_from_dataset_with_abstract,
            inputs=idx_input,
            outputs=[abstract_ds, orig_title_ds, gen_title_ds, sbert_score_ds, sbert_plot_ds]
        )
        btn_clear_ds.click(
            fn=clear_dataset,
            inputs=None,
            outputs=[abstract_ds, orig_title_ds, gen_title_ds, sbert_score_ds, sbert_plot_ds]
        )

    with gr.Tab("Batch Generate from Index Range"):
        range_input = gr.Textbox(placeholder="Enter index range like 0-5", label="Index Range (start-end)")
        with gr.Row():
            btn_submit_batch = gr.Button("Submit", variant="primary")
            btn_clear_batch = gr.Button("Clear")

        batch_output = gr.Textbox(label="Batch Generation Results", lines=20, interactive=False)
        batch_plot = gr.Plot(label="Batch SBERT Similarity Scores")

        btn_submit_batch.click(
            fn=generate_batch_titles,
            inputs=range_input,
            outputs=[batch_output, gr.Textbox(visible=False), gr.Textbox(visible=False), batch_plot]
        )
        btn_clear_batch.click(
            fn=clear_batch,
            inputs=None,
            outputs=[batch_output, gr.Textbox(visible=False), gr.Textbox(visible=False), batch_plot]
        )

    with gr.Tab("From Custom Abstract"):
        abstract_input = gr.Textbox(lines=10, placeholder="Paste abstract here...", label="Abstract")
        with gr.Row():
            btn_submit_ext = gr.Button("Submit", variant="primary")
            btn_clear_ext = gr.Button("Clear")

        gen_title_ext = gr.Textbox(label="Generated Title", interactive=False)

        btn_submit_ext.click(
            fn=generate_from_abstract,
            inputs=abstract_input,
            outputs=[gen_title_ext, gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Plot(visible=False)]
        )
        btn_clear_ext.click(
            fn=clear_abstract,
            inputs=None,
            outputs=[gen_title_ext]
        )

    iface.css = """
    #generated-title-box textarea {
        background-color: #fff9c4;
        font-weight: bold;
        font-size: 1.1em;
        border: 2px solid #fbc02d;
        border-radius: 6px;
        color: #5d4037;
    }
    """

if __name__ == "__main__":
    iface.launch()
