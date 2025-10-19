# code_metrics.py

from typing import List
import re
import ast
import difflib


def normalize_sysml(code: str) -> str:
    """
    Normalize SysMLv2 code:
    - Remove extra spaces
    - Lowercase keywords
    - Strip line endings
    """
    code = code.strip()
    code = re.sub(r"\s+", " ", code)  # collapse multiple spaces
    return code.lower()


def compute_code_bleu(preds: List[str], labels: List[str]) -> float:
    """
    A simple CodeBLEU approximation for SysMLv2.
    Returns a similarity score between 0 and 1.
    """
    scores = []
    for pred, label in zip(preds, labels):
        pred_norm = normalize_sysml(pred)
        label_norm = normalize_sysml(label)

        seq = difflib.SequenceMatcher(None, pred_norm, label_norm)
        score = seq.ratio()  # n-gram similarity approximation
        scores.append(score)

    return sum(scores) / len(scores)


def parse_sysml_ast(code: str) -> List[str]:
    """
    Convert SysMLv2 code to a simple AST representation.
    Each line is treated as a node in this simple tree.
    """
    lines = code.strip().split("\n")
    nodes = [normalize_sysml(line) for line in lines if line.strip()]
    return nodes


def compute_ast_similarity(preds: List[str], labels: List[str]) -> float:
    """
    Compare ASTs line by line using set similarity.
    Returns a score between 0 and 1.
    """
    scores = []
    for pred, label in zip(preds, labels):
        pred_nodes = set(parse_sysml_ast(pred))
        label_nodes = set(parse_sysml_ast(label))
        if not label_nodes:
            scores.append(0.0)
            continue
        intersection = pred_nodes.intersection(label_nodes)
        score = len(intersection) / len(label_nodes)
        scores.append(score)

    return sum(scores) / len(scores)
