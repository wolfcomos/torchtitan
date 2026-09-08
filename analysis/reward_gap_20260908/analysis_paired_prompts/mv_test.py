import sys
from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
CFG=[LatexExtractionConfig(normalization_config=LatexNormalizationConfig(units=True), boxed_match_priority=0, try_extract_without_anchor=False)]
def score(resp, gt):
    gold=parse(gt); pred=parse(resp, extraction_config=CFG, extraction_mode="first_match")
    return float(bool(gold) and verify(gold,pred)), pred
tests=[("work\nAnswer: 225","225"),("work\n**Answer: 37**","37"),("work\n$$\\boxed{225}\n$$\n\nAnswer: 225","225"),("work\nAnswer: $225$","225"),
       ("### Final Answer: 37","37"),("The final answer is 37.\nAnswer: 37","37"),("$$\\boxed{4}$$ earlier ... later $$\\boxed{7}$$\nAnswer: 7","7"),
       ("$$\\boxed{7}$$ ... later $$\\boxed{4}$$\nAnswer: 7","7"),("Answer: \\frac{1}{2}","\\frac{1}{2}"),("Answer: $\\frac{1}{2}$","\\frac{1}{2}"),("Answer: 1/2","\\frac{1}{2}")]
for r,g in tests:
    try: s,p=score(r,g)
    except Exception as e: s,p=("ERR",repr(e)[:80])
    print(f"{s!s:>4}  pred={str(p)[:40]:<40} <- {r!r}")
