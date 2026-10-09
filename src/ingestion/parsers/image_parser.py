"""Image parser: .png .jpg .jpeg .webp  ->  one image Record, plus a copy of the original in assets/.

The record text comes from Gemini vision (caption + printed text + chart/diagram details).
Each analysis is cached beside the image copy, keyed by the file's hash, so re-ingesting an
unchanged image costs no API call (the cache itself lives in vision.analyze_with_cache).
"""
import logging
import shutil
from typing import Callable

from src.core.assets import image_asset_path, relative_to_root
from src.ingestion.file_router import RoutedFile
from src.core.records import Metadata, Record
from src.ingestion.vision import ImageAnalysis, analysis_to_text, analyze_image, analyze_with_cache

log = logging.getLogger(__name__)


def parse_image(
    routed: RoutedFile,
    topic_id: str,
    analyzer: Callable[[bytes, str], ImageAnalysis] = analyze_image,
) -> list[Record]:
    asset = image_asset_path(topic_id, routed.source)
    asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(routed.path, asset)                                  # the untouched original, for the answer step
    analysis = analyze_with_cache(routed.path.read_bytes(), asset.with_name(asset.name + ".analysis.json"), analyzer)

    text = analysis_to_text(analysis, routed.source.rsplit("/", 1)[-1])
    return [Record(text, Metadata(topic_id, routed.source, "image", image_path=relative_to_root(asset)))]
