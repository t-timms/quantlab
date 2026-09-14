from quantlab.gates import GateConfig, GenerationResult, evaluate_generation, is_degenerate

CFG = GateConfig()


def _gen(text, token_ids, finish_reason="stop", pad_token_id: int | None = 0):
    return GenerationResult(
        prompt="p", text=text, token_ids=token_ids, finish_reason=finish_reason, pad_token_id=pad_token_id
    )


def test_healthy_generation_is_not_degenerate():
    gen = _gen("The nth Fibonacci number can be computed with...", list(range(1, 40)))
    assert not is_degenerate(gen, CFG)


def test_zaya1_style_pad_collapse_is_caught():
    # the exact failure mode from the zaya1 project: finish_reason='length',
    # 32/32 token ids all equal to the pad token - empty text, full budget spent
    gen = _gen("", [0] * 32, finish_reason="length", pad_token_id=0)
    assert is_degenerate(gen, CFG)


def test_low_unique_token_repetition_loop_is_caught():
    # the minicpm coding-LoRA regression: repeating the same few tokens
    gen = _gen("((((((((((((((((", [5, 6, 5, 6] * 10)
    assert is_degenerate(gen, CFG)


def test_short_empty_response_is_caught():
    gen = _gen("", [1, 2])
    assert is_degenerate(gen, CFG)


def test_normal_short_but_complete_answer_not_flagged():
    # short is fine as long as it clears min_text_len (5) - matches
    # coherence_check.py's own `len(txt) < 5` threshold exactly, which has no
    # token-count gate. A genuinely tiny answer ("Yes.", 4 chars) DOES still
    # get flagged by design, faithfully carried over - see
    # test_short_empty_response_is_caught for that case.
    gen = _gen("Yes, it does.", [1, 2, 3])
    assert not is_degenerate(gen, CFG)


def test_pad_token_none_does_not_crash_and_does_not_falsely_flag():
    gen = _gen("A coherent multi-token answer here.", list(range(20)), pad_token_id=None)
    assert not is_degenerate(gen, CFG)


def test_evaluate_generation_fails_if_any_result_degenerate():
    healthy = _gen("A real coherent answer with real content.", list(range(15)))
    broken = _gen("", [0] * 20, finish_reason="length")
    verdict = evaluate_generation([healthy, broken], CFG)
    assert verdict.passed is False
    assert verdict.summary() == "FAIL (1/2 degenerate)"


def test_evaluate_generation_passes_when_all_healthy():
    a = _gen("Answer one, coherent and long enough.", list(range(12)))
    b = _gen("Answer two, also coherent.", list(range(10)))
    verdict = evaluate_generation([a, b], CFG)
    assert verdict.passed is True
    assert verdict.summary() == "PASS"
