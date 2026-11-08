"""重放 failed 状态的 Agent 运行：读取失败运行并按原会话重放。"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import get_settings
from backend.app.services.container import build_services
from backend.app.services.query.run_replay import (
    list_failed_runs,
    replay_failed_run,
)


async def _never_disconnected() -> bool:
    """重放脚本不响应客户端断开，保持请求完整执行。"""
    return False


async def _run(
    *,
    user_id: str,
    run_id: str,
    limit: int,
    dry_run: bool,
) -> int:
    """列出并重放失败运行，只输出运行编号与状态，不输出消息正文。"""
    settings = get_settings()
    services = build_services(settings)
    store = services.session_store_for(user_id)
    if run_id:
        candidates = [store.get_agent_run(run_id)]
        candidates = [run for run in candidates if run is not None]
    else:
        candidates = await list_failed_runs(store, limit=limit)
    if not candidates:
        print("没有需要重放的失败运行")
        return 0
    if dry_run:
        for run in candidates:
            print(
                f"[dry-run] run_id={run['run_id']} "
                f"session_id={run['session_id']}"
            )
        return 0
    chat_service = services.chat_service_for(user_id)
    for run in candidates:
        result = await replay_failed_run(
            store,
            str(run["run_id"]),
            stream_fn=chat_service.stream,
            is_disconnected=_never_disconnected,
        )
        skipped = "skipped" if result.get("skipped") else "replayed"
        print(
            f"[{skipped}] run_id={result.get('run_id')} "
            f"status={result.get('status') or result.get('reason')} "
            f"events={result.get('events') or 0}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造带安全默认值的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="读取 agent_runs 中 failed 状态的运行并按原会话重放。"
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("REPLAY_USER_ID", "admin"),
        help="要重放的用户 ID；默认读取 REPLAY_USER_ID 或 admin",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="只重放指定运行 ID；为空时重放全部失败运行",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="批量重放最多处理多少条失败运行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出失败运行，不真正执行重放",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只输出运行编号与状态摘要。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(
        _run(
            user_id=str(args.user_id),
            run_id=str(args.run_id).strip(),
            limit=max(1, int(args.limit)),
            dry_run=bool(args.dry_run),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
