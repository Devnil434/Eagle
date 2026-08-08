from __future__ import annotations
import logging

from fastapi import APIRouter

logger = logging.getLogger("eagle.snapshot")

router = APIRouter()


@router.get("", tags=["snapshot"])
def ping():
    return {"ok": True}
