from flask import Flask, request, jsonify
import whisperx
from dotenv import load_dotenv
import torch

def transcribe_audio(audio_file):
    
    torch.cuda.empty_cache()
    device = "cuda" 
    # audio_file = "c7d56d00-b760-4a39-9d2b-2a3e011d1101.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    language = 'fr'
    model_name = 'large-v2'
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_name, device, compute_type=compute_type,language=language)
    HF_TOKEN = "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"

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio,min_speakers=2,max_speakers=3)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs
    return result['segments']
app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    file_path = data.get("file_path")

    if not file_path:
        return jsonify({"error": "File path not provided"}), 400

    # Run WhisperX processing here (assuming `whisperx.transcribe` method for example)
    result = transcribe_audio(file_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)