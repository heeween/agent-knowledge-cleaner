from http.client import REQUEST_TIMEOUT
import os
import time

from openai import OpenAI


MODEL = "qwen3.5-flash"

client = OpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)


def test(name, **extra):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": (
                    "请判断下面的问题属于哪个CRM模块。"
                    "只返回JSON。\n\n"
                    "问题：为什么客户的保养提醒没有出现？"
                ),
            }
        ],

        temperature=0,

        response_format={
            "type": "json_object"
        },

        extra_body={
            "enable_thinking": False
        },

        **extra,
    )

    elapsed = time.time() - start

    message = response.choices[0].message

    print(f"耗时: {elapsed:.2f}s")

    print(
        "content:",
        message.content
    )

    if response.usage:

        print(
            "prompt_tokens:",
            response.usage.prompt_tokens
        )

        print(
            "completion_tokens:",
            response.usage.completion_tokens
        )

        print(
            "total_tokens:",
            response.usage.total_tokens
        )

        try:
            print(
                "details:",
                response.usage
                .completion_tokens_details
            )
        except Exception:
            pass

    reasoning = getattr(
        message,
        "reasoning_content",
        None,
    )

    if reasoning:
        print(
            "reasoning length:",
            len(reasoning)
        )

        print(
            "reasoning preview:",
            reasoning[:500]
        )


test("默认参数")

