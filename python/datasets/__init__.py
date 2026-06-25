"""Dataset readers / writers shared across the limited pipeline.

The writers live next to the data they produce (drafts in
``python/draft/dataset.py``, gameplay in ``python/gameplay/dataset_writer.py``)
so each subpackage stays self-contained. *This* module owns the
**reader** side: opening a directory of parquet files and exposing
pandas/iterator APIs that the training and analytics modules consume.

For now everything is parquet — msgpack/npz support can be added when
the storage backend diverges per artefact type.
"""

from .reader import DatasetReader, list_artifacts

__all__ = ["DatasetReader", "list_artifacts"]
