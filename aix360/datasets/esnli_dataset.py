import os

from functools import lru_cache
from typing import Dict

# helper function that returns a specific
# sentence pair example from the e-SNLI dataset
@lru_cache(maxsize=120)
def _example(file: str, id: str) -> Dict:
    import json
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            try:
                line = f.readline().strip()

                if line == '':
                    raise EOFError

                d = json.loads(line)
                if d['docid'] == id:
                    return d
            except EOFError:
                raise RuntimeError(f"example {id} not found")

class eSNLIDataset:
    """
    The e-SNLI dataset [#]_ contains pairs of sentences 
    each accompanied by human-rationale annotations 
    as to which words are in each pairs are most
    important for matching.

    The sentence pairs are from the Stanford Natural
    Language Inference dataset with labels that indicate
    if the sentence pair is a logical entailment,
    contradiction or neutral.

    References:
        .. [#] `Camburu, Oana-Maria, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom, 
          “E-SNLI: Natural Language Inference with Natural Language Explanations.”,
          2018
          <https://arxiv.org/abs/1812.01193>`_
    """

    def __init__(self):
        self._dirpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data','esnli_data'
        )

        self._cache_doc = {}
        
    def get_example(self, example_id: str) -> Dict:
        """
        Return an e-SNLI example.

        The example_id indexes the "docs.jsonl" file of the downloaded dataset.

        Args:
            example_id (str): the example index.

        Returns:
            e-SNLI example in dictionary form.
        """
        return _example(
            os.path.join(
                self._dirpath,
                'docs.jsonl'
            ),
            example_id,
        )

