# DSTA BrainHack TIL-AI 2025

### Team: Hackoholics

### 👥 Team Members

| Name                                        | Role    |
| ------------------------------------------- | ------- |
| [richierx](https://github.com/richierx)     | RL, ASR |
| [w3joe](https://github.com/w3joe)           | RL      |
| [cjiayu03](https://github.com/cjiayu03)     | OCR     |
| [electrum21](https://github.com/electrum21) | CV      |

We managed to qualify with 13th place in the Novice category, but was defeated in the semi-finals.

---

## Contents

- [Introduction](#introduction)
  - [RL](#rl)
  - [ASR](#asr)
  - [OCR](#ocr)
  - [CV](#cv)
  - [Get started](#get-started)
  - [Understanding this repo](#understanding-this-repo)
  - [Build, test, and submit](#build-test-and-submit)

## RL

In a multi-agent environment, the Scout must navigate a dynamic map, avoid capture by the Guards, collect Recon Points, and complete challenges at fixed locations. Meanwhile, the Guards are trained to patrol and intercept the Scout efficiently.

**Solution:**  
We implemented a hybrid approach: supervised learning and imitation learning were used to teach the Scout to move strategically, while the Guards were trained using a CNN-based DQN agent with experience generated via BFS pathfinding.

📄 [Read more about RL →](./rl/README.md)

## ASR

The challenge is to transcribe spoken content from noisy audio recordings. The model must handle speech under varying conditions and accurately produce a written transcript despite distortions or interference.

**Solution:**  
We fine-tuned OpenAI's Whisper Small model on domain-specific and noise-augmented data to improve transcription quality in adverse environments.

📄 [Read more about ASR →](./asr/README.md)

## OCR

The goal is to extract structured text from images of scanned documents. These documents may contain varying fonts, alignments, and noise artifacts that reduce legibility.

**Solution:**  
We fine-tuned the Tesseract OCR engine on our dataset, optimizing preprocessing steps to improve character segmentation and recognition accuracy.

📄 [Read more about OCR →](./ocr/README.md)

## CV

Detect and classify specific target objects within diverse images. Each image may contain multiple instances—or none—of the objects in question, requiring precise localization and accurate classification.

**Solution:**  
We employed strong data augmentation techniques and used RT-DETRv2-M (a real-time detection transformer) for fast and accurate object detection and classification.

📄 [Read more about CV →](./cv/README.md)

## Get started

Here's a quick overview of the initial setup instructions. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

Use this repository as a template to create your own, and clone it into your Vertex AI workbench. You'll want to keep your repository private, so you'll need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

You'll also need to initialize the Git submodules:

```bash
git submodule update --init
```

This repository requires Python 3.12 or newer to work. Use `conda` to create a virtual environment with a Python version of at least 3.12.

```bash
conda create --name til python=3.12
conda activate til
```

Finally, install the development dependencies into your newly created virtual environment.

```bash
pip install -r requirements-dev.txt
```

## Understanding this repo

There's a subdirectory for each challenge: [`asr/`](/asr), [`cv/`](/cv), [`ocr/`](/ocr/), and [`rl/`](/rl). Each contains:

- A `src/` directory, where your code lives.
  - `*_manager.py`, which manages your model. This is where your inference and computation takes place.
  - `*_server.py`, which runs a local web server that talks to the rest of the competition infrastructure.
- `Dockerfile`, which is used to build your Docker image for each model.
- `requirements.txt`, which lists the dependencies you need to have bundled into your Docker image.
- `README.md`, which contains specifications for the format of each challenge.

You'll also find a final subdirectory, [`test/`](/test). This contains tools to test and score your model locally.

There are also two Git submodules, `til-25-finals` and `til-25-environment`. `finals` contains code that will be pulled into your repo for Semifinals and Finals. `environment` contains the `til_environment` package, which will help you train and test your RL model, and is installed by pip during setup. Don't delete or modify the contents of `til-25-finals/`, `til-25-environment/`, or `.gitmodules`.

## Build, test, and submit

Submitting your model for evaluation is simple: just build your Docker image, test it, and submit. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

You'll first want to `cd` into the directory you want to build. Then, build the image using Docker, following the naming scheme below. You should then run and test your model before using `til submit` to submit your image for evaluation.

```bash

# cd into the directory. For example, `cd ./asr/`
cd CHALLENGE

# Build your image. Remember the . at the end.
docker build -t TEAM_ID-CHALLENGE:TAG .

# Optionally, you can run your model locally to test it.
docker run -p PORT --gpus all -d IMAGE_NAME:TAG
python test/test_CHALLENGE.py

# Push it for submission
til submit TEAM_ID-CHALLENGE:TAG
```
