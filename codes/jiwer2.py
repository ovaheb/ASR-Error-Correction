from collections import defaultdict
from typing import List, Optional, Union

from jiwer.process import AlignmentChunk, CharacterOutput, WordOutput

__all__ = ["collect_error_counts", "visualize_alignment", "visualize_error_counts"]

def visualize_error_counts(output, show_substitutions=True, show_insertions=True, show_deletions=True, top_k=None):
    s, i, d = collect_error_counts(output)

    def build_list(errors: dict):
        if len(errors) == 0:
            return "none"

        keys = [k for k in errors.keys()]
        keys = sorted(keys, reverse=True, key=lambda k: errors[k])

        if top_k is not None:
            keys = keys[:top_k]

        # we get the maximum length of all words to nicely pad output
        ln = max(len(k) if isinstance(k, str) else max(len(e) for e in k) for k in keys)

        # here we construct the string
        build = ""

        for count, (k, v) in enumerate(
            sorted(errors.items(), key=lambda tpl: tpl[1], reverse=True)
        ):
            if top_k is not None and count >= top_k:
                break

            if isinstance(k, tuple):
                build += f"{k[0]: <{ln}} --> {k[1]:<{ln}} = {v}x\n"
            else:
                build += f"{k:<{ln}} = {v}x\n"

        return build

    output = ""

    if show_substitutions:
        if output != "":
            output += "\n"
        output += "=== SUBSTITUTIONS ===\n"
        output += build_list(s)

    if show_insertions:
        if output != "":
            output += "\n"
        output += "=== INSERTIONS ===\n"
        output += build_list(i)

    if show_deletions:
        if output != "":
            output += "\n"
        output += "=== DELETIONS ===\n"
        output += build_list(d)

    if output[-1:] == "\n":
        output = output[:-1]

    return output

def collect_error_counts(output: Union[WordOutput, CharacterOutput]):
    substitutions = defaultdict(lambda: 0)
    insertions = defaultdict(lambda: 0)
    deletions = defaultdict(lambda: 0)

    for idx, sentence_chunks in enumerate(output.alignments):
        ref = output.references[idx]
        hyp = output.hypotheses[idx]
        sep = " " if isinstance(output, WordOutput) else ""

        for chunk in sentence_chunks:
            if chunk.type == "insert":
                inserted = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                insertions[inserted] += 1
            if chunk.type == "delete":
                deleted = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                deletions[deleted] += 1
            if chunk.type == "substitute":
                replaced = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                by = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                substitutions[(replaced, by)] += 1

    return substitutions, insertions, deletions