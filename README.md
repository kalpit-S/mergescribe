# MergeScribe 🎤

**MergeScribe** is a "Local God Mode" transcription tool designed to reduce the bandwidth between your mind and your machine. It combines local speed with cloud intelligence and multi-microphone redundancy to deliver the fastest, most accurate voice typing experience possible on macOS.

![MergeScribe Demo](demos/transcription_demo.gif)

## 🚀 Key Features

### 1. Local God Mode (Multi-Mic Redundancy) 🎙️🎙️🎙️
Why rely on one microphone? MergeScribe can listen to **all your microphones simultaneously**.
- **HyperX SoloCast** (High Fidelity)
- **MacBook Pro Built-in** (Room Tone)
- **AirPods Pro** (Close Talk)

It processes all streams in parallel using local AI (Parakeet) or Cloud AI (Groq), then uses an LLM to "vote" on the best transcription. If one mic muffles a word, another will catch it.

### 2. Blazing Fast "Hedged" Correction ⚡
Latency is the enemy.
- **Parallel Processing:** All audio streams are transcribed instantly.
- **Hedged Requests:** We fire **two** identical correction requests to the LLM API simultaneously. The first one to return a byte wins. This smooths out network jitter.
- **Fast Path:** If all your microphones agree on a short command (< 15 words), we **skip the LLM entirely** and type instantly. Zero cost, zero wait.

### 3. Context Awareness 🧠
MergeScribe knows what you are doing.
- **App Detection:** If you are in VS Code, it biases towards Python/JS syntax. If you are in Slack, it prefers casual conversational style.
- **History:** It remembers the last few sentences to maintain context ("it" refers to the previous noun).

### 4. Text Editing Mode ✏️
Highlight any text, press the hotkey, and say "Change Parachute to Parakeet."
- The app grabs the selected text.
- Applies your voice command using a specialized LLM prompt.
- Replaces the text instantly.

---

## 🛠️ Installation

1.  **Clone & Setup:**
    ```bash
    git clone https://github.com/yourname/mergescribe.git
    cd mergescribe
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Install Parakeet (Apple Silicon Optimized):**
    ```bash
    pip install parakeet-mlx
    ```

3.  **Configure:**
    - Create a `.env` file with your keys:
        ```env
        GROQ_API_KEY=gsk_...
        OPENROUTER_API_KEY=sk-or-...
        ```
    - Or use the UI Settings menu.

4.  **Run:**
    ```bash
    python main.py
    ```

---

## ⚙️ Configuration

Open the **Settings** menu via the 🎙️ icon in the menu bar.

*   **General:** Set your Trigger Key (default: Right Option).
*   **Audio:** Check **ALL** the microphones you want to use.
    *   *Recommendation:* Select your high-quality USB mic AND your built-in MacBook mic.
*   **Models:** Choose `parakeet_mlx` (Local, Free, Fast) or `groq_whisper` (Cloud, Fast).
*   **Context:** Enable "Application Context" for smarter results.

---

## 🗺️ Roadmap & Philosophy

We are building a self-improving system. See [ROADMAP.md](ROADMAP.md) for details on:
- **Data Flywheel:** Logging recordings to build a personal fine-tuning dataset.
- **RLHF:** Learning from your manual edits to fix recurring typos automatically.
- **DSPy:** Self-optimizing prompts that rewrite themselves based on your corrections.

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User Speaks] --> Mic1[Mic 1: HyperX]
    User --> Mic2[Mic 2: MacBook]
    User --> Mic3[Mic 3: AirPods]
    
    Mic1 -->|Stream| LocalAI[Parakeet MLX (v3)]
    Mic2 -->|Stream| LocalAI
    Mic3 -->|Stream| LocalAI
    
    LocalAI -->|Text A| Consensus{Consensus?}
    LocalAI -->|Text B| Consensus
    LocalAI -->|Text C| Consensus
    
    Consensus -->|Yes (Fast Path)| Typer[Type Output]
    Consensus -->|No| LLM[LLM Correction]
    
    LLM -->|Hedged Req 1| Race{Race}
    LLM -->|Hedged Req 2| Race
    
    Race -->|Winner| Typer
```

License: MIT