"""ComfyUIJobsAgent — delivers finished ComfyUI jobs to the inbox.

The ``comfyui`` command queues a workflow and saves a job record; every
POLL_SECONDS this agent asks ComfyUI's ``/history`` about each one, downloads
the result through ``/view`` and posts it to the user's inbox as an attachment
with a push (or posts a readable failure: out of memory, a missing model, the
server gone, a 30-minute timeout). See comfyui_shared.jobs. With no jobs saved
a tick makes no request at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from jarvis_command_sdk import AgentSchedule, Alert, IJarvisAgent, IJarvisSecret

from comfyui_shared.jobs import poll_jobs
from comfyui_shared.log import logger

POLL_SECONDS = 20


class ComfyUIJobsAgent(IJarvisAgent):
    """Polls queued ComfyUI jobs and posts each result to the inbox."""

    @property
    def name(self) -> str:
        return "comfyui_jobs"

    @property
    def description(self) -> str:
        return "Delivers finished ComfyUI videos and pictures to the inbox."

    @property
    def schedule(self) -> AgentSchedule:
        return AgentSchedule(interval_seconds=POLL_SECONDS, run_on_startup=True)

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        # The command declares COMFYUI_URL / COMFYUI_API_KEY; the agent reads the same
        # integration secrets and does nothing until a job exists.
        return []

    @property
    def include_in_context(self) -> bool:
        return False

    def get_context_data(self) -> Dict[str, Any]:
        return {}

    def get_alerts(self) -> List[Alert]:
        return []

    async def run(self) -> None:
        self._tick(datetime.now(timezone.utc))

    # Sync core so it's deterministically testable (run() just supplies "now").
    def _tick(self, now: datetime) -> None:
        try:
            poll_jobs(now)
        except Exception as exc:  # never let one bad run kill the scheduler
            logger.error("comfyui agent: poll failed", error=str(exc))
