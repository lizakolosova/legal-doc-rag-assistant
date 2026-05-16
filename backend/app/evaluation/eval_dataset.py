import json
import logging
from pathlib import Path

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class GoldenQA(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant_sources: list[dict]


def load_golden_dataset(path: Path | None = None) -> list[GoldenQA]:
    """Load the golden Q&A dataset from a JSON file.

    Args:
        path: Path to the JSON file; defaults to eval_data/legal_qa_golden.json.

    Returns:
        List of GoldenQA objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist at the resolved path.
    """
    resolved = path if path is not None else Path(settings.eval_data_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Golden dataset not found at '{resolved}'. "
            "Create eval_data/legal_qa_golden.json or pass an explicit path.")

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    dataset = [GoldenQA(**item) for item in raw]
    logger.info("Loaded %d golden Q&A pairs from %s", len(dataset), resolved)
    return dataset