"""Actor-SFT rollout: no generation — tokenize prompt+response, stash activation.

Pattern follows miles/rollout/sft_rollout.py. The data_buffer yields Samples
whose .prompt is a list[dict] (from NLADataSource, <INJECT>→㊗ already substituted)
and whose .metadata["response"] is the <explanation>...</explanation> string.

Loss-mask construction is the head/full prefix split:

    head = apply_chat_template(user_msgs, add_generation_prompt=True, **kwargs)
    full = apply_chat_template(user_msgs + [assistant], **kwargs)
    response tokens = full[len(head):]         (loss_mask = 1)

This makes SFT train on EXACTLY the token positions RL will generate —
nla_generate builds its rollout prompt with the same apply_chat_template
call and the same --apply-chat-template-kwargs. For thinking-capable models
(Qwen3/3.5), pass --apply-chat-template-kwargs '{"enable_thinking": false}':
the template then appends the empty '<think>\\n\\n</think>\\n\\n' block to the
GENERATION PROMPT (head), and renders the assistant turn with the identical
empty block — so the think scaffold lands in the masked prompt on both sides
and the model is trained/rolled-out with thinking disabled consistently.

The prefix property (full[:len(head)] == head) is asserted per sample — it
catches template drift AND BPE merges across the head/response boundary.
"""

import torch

from miles.utils.processing_utils import load_tokenizer

from nla.schema import MM_ACTIVATION_KEY


_TOKENIZER = None
_SAMPLE_PRINTED = False


def _tokenize_with_loss_mask(tokenizer, messages, chat_kwargs):
    """(token_ids, response_length) via the head/full prefix split."""
    assert messages[-1]["role"] == "assistant", messages[-1]["role"]
    # The prefix split trains ONLY the final assistant message. An earlier
    # assistant turn would silently become pure context (no loss) — refuse
    # rather than under-train. (--loss-mask-type is also ignored here; NLA
    # data is single-user-turn by construction so neither should ever fire.)
    assert not any(m["role"] == "assistant" for m in messages[:-1]), (
        "multi-turn prompt with earlier assistant messages — the prefix-split "
        "loss mask only supervises the final turn; use a multi-turn mask "
        "generator instead."
    )
    head = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        return_dict=False, **chat_kwargs,
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=False, **chat_kwargs,
    )
    assert len(full) > len(head) and full[: len(head)] == head, (
        f"chat-template head is not a token-prefix of the full render — "
        f"either the template inserts content between the generation prompt "
        f"and the assistant text (check --apply-chat-template-kwargs, e.g. "
        f"enable_thinking), or BPE merged across the boundary.\n"
        f"head tail: {tokenizer.convert_ids_to_tokens(head[-8:])}\n"
        f"full there: {tokenizer.convert_ids_to_tokens(full[max(0,len(head)-8):len(head)+4])}"
    )
    return full, len(full) - len(head)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _TOKENIZER, _SAMPLE_PRINTED
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    chat_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for group in samples:
        (sample,) = group
        messages = sample.prompt
        assert isinstance(messages, list), (
            f"actor SFT requires list[dict] prompt (got {type(messages).__name__}). "
            f"NLADataSource must use apply_chat_template=False."
        )
        response = sample.metadata["response"]
        messages = messages + [{"role": "assistant", "content": response}]

        token_ids, response_length = _tokenize_with_loss_mask(
            _TOKENIZER, messages, chat_kwargs
        )

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0.0
        sample.loss_mask = [1] * response_length

        if not _SAMPLE_PRINTED:
            _SAMPLE_PRINTED = True
            print(
                f"[NLA sft_actor] first sample: {len(token_ids)} tokens, "
                f"response={response_length}, prompt tail="
                f"{_TOKENIZER.convert_ids_to_tokens(token_ids[len(token_ids)-response_length-6:len(token_ids)-response_length])!r} "
                f"response head={_TOKENIZER.convert_ids_to_tokens(token_ids[len(token_ids)-response_length:len(token_ids)-response_length+6])!r}",
                flush=True,
            )

        activation = torch.tensor(
            sample.metadata["activation_vector"], dtype=torch.float32
        ).view(1, -1)
        sample.multimodal_train_inputs = {MM_ACTIVATION_KEY: activation}

    return samples
