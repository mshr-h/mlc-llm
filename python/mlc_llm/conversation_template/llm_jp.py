"""LLM-jp default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

_LLM_JP_4_SYSTEM_MESSAGE = (
    "You are LLM-jp-4, a large language model trained by LLM-jp.\n"
    "Knowledge cutoff: 2025-12\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)


def _get_llm_jp_4_harmony_template(name: str, system_message: str) -> Conversation:
    return Conversation(
        name=name,
        system_template=f"<|start|>system<|message|>{MessagePlaceholders.SYSTEM.value}<|end|>",
        system_message=system_message,
        roles={
            "user": "<|start|>user<|message|>",
            "assistant": "<|start|>assistant",
        },
        seps=["<|end|>"],
        role_templates={
            "user": MessagePlaceholders.USER.value,
            "assistant": f"<|channel|>final<|message|>{MessagePlaceholders.ASSISTANT.value}",
        },
        role_content_sep="",
        role_empty_sep="",
        stop_str=["<|return|>"],
        stop_token_ids=[],
        add_role_after_system_message=True,
    )


# LLM-jp instruct
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="llm-jp",
        system_template=f"{MessagePlaceholders.SYSTEM.value}",
        system_message="以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
        roles={
            "user": "\n\n### 指示:",
            "assistant": "\n\n### 応答:",
        },
        seps=["", "</s>"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=[],
        stop_token_ids=[2],  # eos_token_id
        system_prefix_token_ids=[1],  # bos_token_id (<s>)
        add_role_after_system_message=True,
    )
)

# LLM-jp-4 Harmony normal chat.
# The base llm-jp-4 model should use the generic "LM" template.
ConvTemplateRegistry.register_conv_template(
    _get_llm_jp_4_harmony_template(
        name="llm_jp_4_harmony",
        system_message=_LLM_JP_4_SYSTEM_MESSAGE,
    )
)

# LLM-jp-4 thinking uses the same Harmony framing, with the reasoning-effort
# line that appears in the local thinking model's chat_template.jinja.
ConvTemplateRegistry.register_conv_template(
    _get_llm_jp_4_harmony_template(
        name="llm_jp_4_harmony_thinking",
        system_message=_LLM_JP_4_SYSTEM_MESSAGE.replace(
            "\n\n# Valid channels:",
            "\n\nReasoning: medium\n\n# Valid channels:",
        ),
    )
)
