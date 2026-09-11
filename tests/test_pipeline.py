from whisper_asr_pipeline import WhisperASRPipeline, word_error_rate


def test_wer():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello") == 0.5


def test_wer_ignores_case_and_punctuation_but_not_spelling():
    reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD"
    hypothesis = "Mr. Quilter is the apostle of the middle classes, and we are glad."
    assert word_error_rate("hello world", "Hello, world.") == 0.0
    assert word_error_rate("don't stop", "Don't stop!") == 0.0
    assert word_error_rate("don't stop", "Don\u2019t stop") == 0.0
    assert word_error_rate("stra\u00dfe", "STRASSE") == 0.0
    assert word_error_rate("well-known", "well known") == 2.0  # one reference token, two errors
    assert word_error_rate(reference, hypothesis) == 1 / 13


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
