# scripts/export_finetuning_dataset.py

import asyncio
import argparse
import logging
import sys
import os
from Eagle.libs.config import settings
from redis import asyncio as aioredis
from apps.backend.services.feedback_collector import FeedbackCollector
from services.data_collection.exporter import LLaVAExporter
from libs.config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main(redis_url: str, output_dir: str, output_filename: str = None):
    """Orchestrate Redis connection, feedback collection, and dataset export."""
    
    redis_client = None
    try:
        # Initialize Redis connection
        redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        logger.info(f"Connected to Redis: {redis_url}")
        
        # Create feedback collector with Redis client
        collector = FeedbackCollector(redis_client)
        
        # Create exporter with output directory
        exporter = LLaVAExporter(collector, output_dir=output_dir)
        
        # Export with metadata (JSONL + stats file)
        result = await exporter.export_with_metadata(output_filename)
        
        # Print results to stdout
        print("\n" + "="*60)
        print("  Export completed successfully")
        print(f"  JSONL file: {result['jsonl_path']}")
        print(f"  Metadata file: {result['metadata_path']}")
        print(f"  Records exported: {result['record_count']}")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        print(f"\n  Export failed: {e}\n", file=sys.stderr)
        return 1
        
    finally:
        # Cleanup Redis connection
        if redis_client:
            await redis_client.aclose()
            logger.info("Redis connection closed")

def parse_args():
    """Parse command-line arguments for export configuration."""

    parser = argparse.ArgumentParser(
        description="Export feedback dataset in LLaVA fine-tuning format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" """
    )
    
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", settings.REDIS_URL),
        help="Redis connection URL (default: env REDIS_URL or localhost:6379)"
    )
    parser.add_argument(
        "--output-dir",
        default="./datasets",
        help="Output directory for JSONL and metadata files (default: ./datasets)"
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Custom output filename (default: finetuning_dataset_{timestamp}.jsonl)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run async main function
    exit_code = asyncio.run(main(
        redis_url=args.redis_url,
        output_dir=args.output_dir,
        output_filename=args.filename
    ))
    
    sys.exit(exit_code)