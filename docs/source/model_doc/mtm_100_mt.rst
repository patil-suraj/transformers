.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

M2M100
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M2M100 model was proposed in `Beyond English-Centric Multilingual Machine Translation <https://arxiv.org/abs/2010.11125>`__ by Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav
Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin, Facebook AI, LORIA.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*Existing work in translation demonstrated the potential of massively multilingual machine
translation by training a single model able to translate between any pair of languages.
However, much of this work is English-Centric by training only on data which was translated
from or to English. While this is supported by large sources of training data, it does
not reflect translation needs worldwide. In this work, we create a true Many-to-Many
multilingual translation model that can translate directly between any pair of 100 languages.
We build and open source a training dataset that covers thousands of language directions
with supervised data, created through large-scale mining. Then, we explore how to effectively
increase model capacity through a combination of dense scaling and language-specific sparse
parameters to create high quality models. Our focus on non-English-Centric models brings
gains of more than 10 BLEU when directly translating between non-English directions while
performing competitively to the best single systems of WMT.We open-source our scripts
so that others may reproduce the data, evaluation, and final M2M-100 model `here <https://github.com/pytorch/fairseq/tree/master/examples/m2m_100>`__*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

M2M100Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.M2M100Config
    :members:


M2M100Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.M2M100Tokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


M2M100Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.M2M100Model
    :members: forward


M2M100ForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.M2M100ForConditionalGeneration
    :members: forward


