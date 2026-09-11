from whisper_asr_pipeline import WhisperASRPipeline, word_error_rate


def test_wer():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello") == 0.5


def test_normalized_output():
    calls = {}

    def fake(audio, **kwargs):
        calls.update(kwargs)
        return {"text": " hello world ", "chunks": []}

    pipeline = WhisperASRPipeline(fake, "cpu")
    result = pipeline.transcribe(
        {"array": [0.0], "sampling_rate": 16_000},
        language="en",
    )
    assert result["text"] == "hello world"
    assert result["model_revision"]
    assert calls["generate_kwargs"]["language"] == "en"
