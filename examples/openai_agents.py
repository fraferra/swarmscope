"""Instrumenting a homegrown OpenAI loop. Needs OPENAI_API_KEY and the ``openai`` extra.

    python examples/openai_agents.py
    swarmscope inspect && swarmscope waste
"""
import os

import swarmscope as ss
from swarmscope.adapters.openai_transport import instrument_openai

sdk = ss.init("sqlite:///swarmscope.db")
instrument_openai(sdk)  # every OpenAI() client created from now on is traced at the transport layer

from openai import OpenAI  # noqa: E402

client = OpenAI()


@sdk.agent(role="solver", group="math")
def solve(question: str) -> str:
    hit = sdk.claim(f"solve: {question}", kind="approach")
    if hit.suggest_skip:
        hit.suppress("another agent is on it")
        return ""
    r = client.chat.completions.create(model="gpt-4o-mini",
                                       messages=[{"role": "user", "content": question}])
    answer = r.choices[0].message.content or ""
    art = sdk.artifact(answer, kind="answer")
    # Replace with a real verifier; a verdict is what makes waste/ablation meaningful.
    sdk.verdict(art, status="accepted" if answer.strip() else "rejected", source="verifier")
    return answer


@sdk.consolidator
def pick(contributions):
    return next((c.value for c in contributions if c.value), None)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    with sdk.run("openai-demo"):
        outs = [(f"q{i}", solve("What is 17*23? Answer with the number only.")) for i in range(3)]
        print(pick(outs))
    sdk.close()
