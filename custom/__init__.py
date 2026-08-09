"""Custom LUNAR use case — a copy-paste template.

Copy this ``custom/`` folder, rename it, and replace the ``Custom*`` classes with
your own implementations:

* :class:`~custom.custom_sut.CustomSUT`               — your System Under Test.
* :class:`~custom.custom_fitness.CustomFitness`       — what a *failure* is.
* :class:`~custom.custom_generator.CustomUtteranceGenerator` — how test inputs are built.
* :class:`~custom.custom_models.CustomContentInput` / :class:`~custom.custom_models.CustomOutputModel`.

Out of the box everything is **mocked and runs fully offline** (no LLM, no
server): the generator builds templated utterances and the SUT returns a random
output model. Run it with::

    python -m custom.main --preset test

See ``README.md`` for the full walkthrough and the hyperparameter presets.
"""
