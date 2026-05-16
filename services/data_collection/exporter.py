import json
import logging
import os
import aiofiles
from datetime import datetime, timezone
from pathlib import Path
from libs.schemas.feedback import LLaVAConversation, Conversation, FeedbackRecord
from apps.backend.services.feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)

class LLaVAExporter:
    def __init__(self, collector: FeedbackCollector, output_dir: str = "./datasets"):
        """Initialize exporter with feedback collector and output directory."""
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def format_to_llava(self, record: FeedbackRecord) -> LLaVAConversation:
        """Convert FeedbackRecord to LLaVA conversation format with human/gpt roles."""
        
        human_msg = f"Original prediction: {record.original_label}. Captions: {' → '.join(record.caption_sequence)}"
        
        # Second turn: gpt (model) accepts human correction
        gpt_msg = f"Corrected to: {record.human_label}. Note: {record.human_note if record.human_note else 'N/A'}"
        
        # Image filename derived from alert_id
        image_filename = f"frame_{record.alert_id}.jpg"
        
        return LLaVAConversation(
            image=image_filename,
            conversations=[
                Conversation(from_="human", value=human_msg),
                Conversation(from_="gpt", value=gpt_msg)
            ]
        )

    async def export_to_jsonl(self, output_filename: str = None) -> tuple:
        """Stream feedback to JSONL format with async file I/O, return (path, record_count)."""
        if not output_filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_filename = f"finetuning_dataset_{timestamp}.jsonl"
        
        output_path = self.output_dir / output_filename
        record_count = 0
        
        try:
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                async for record in self.collector.get_all_feedback():
                    llava_record = await self.format_to_llava(record)
                    await f.write(llava_record.model_dump_json() + '\n')
                    record_count += 1
            
            logger.info(f"Exported {record_count} records to {output_path}")
            return (str(output_path), record_count)
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            if output_path.exists():
                output_path.unlink()
            raise

    async def export_with_metadata(self, output_filename: str = None) -> dict:
        """Export JSONL and create metadata file with dataset stats."""
        jsonl_path, record_count = await self.export_to_jsonl(output_filename)
        
        # Metadata: dataset info for training pipeline
        metadata = {
            "dataset_file": os.path.basename(jsonl_path),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "record_count": record_count,
            "format": "LLaVA conversation",
            "image_dir": "frames/"
        }
        
        metadata_path = self.output_dir / f"{Path(jsonl_path).stem}_metadata.json"
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"Metadata written to {metadata_path}")
        
        return {
            "jsonl_path": jsonl_path,
            "metadata_path": str(metadata_path),
            "record_count": record_count
        }