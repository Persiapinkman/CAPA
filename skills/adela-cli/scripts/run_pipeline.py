import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from run_cli import (
    benchmark_add,
    benchmark_info,
    benchmark_list,
    deployment_add,
    deployment_info,
    deployment_list,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "references"))
EVAL_TYPE_MAP = {
    0: "normal_precision",
    1: "normal_performance",
}
EVAL_CONFIG_MAP = {
    0: os.path.join(REFERENCE_DIR, "accuracy_eval.json"),
    1: os.path.join(REFERENCE_DIR, "speed_eval.json"),
}
# 评测提交用的临时 JSON：固定目录，成功后不删除，便于排障与复用路径
ADELA_EVAL_CONFIG_DIR = "/tmp/adela-cli"
# 部署轮询：每 30 秒一次，最多 30 次（约 15 分钟内）
# 评测结果轮询：每 30 秒一次，最多 30 次（约 15 分钟内）
MAX_POLL_COUNT = 30
DEPLOYMENT_POLL_INTERVAL_SECONDS = 30
BENCHMARK_POLL_INTERVAL_SECONDS = 30


def _is_quant_platform(platform: str) -> bool:
    p = platform.lower()
    return "int4" in p or "int8" in p


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_deployment_id(data: Dict[str, Any]) -> Optional[int]:
    for key in ("id", "did", "deployment_id"):
        if key in data:
            parsed = _safe_int(data.get(key))
            if parsed is not None:
                return parsed
    return None


def _extract_benchmark_id(data: Dict[str, Any]) -> Optional[int]:
    for key in ("id", "bid", "benchmark_id"):
        if key in data:
            parsed = _safe_int(data.get(key))
            if parsed is not None:
                return parsed
    return None


def _find_quant_dataset_id_from_info(info: Any) -> Optional[int]:
    if isinstance(info, dict):
        for k, v in info.items():
            key = k.lower()
            if "quant" in key and "dataset" in key and "id" in key:
                parsed = _safe_int(v)
                if parsed is not None:
                    return parsed
            nested = _find_quant_dataset_id_from_info(v)
            if nested is not None:
                return nested
    elif isinstance(info, list):
        for item in info:
            nested = _find_quant_dataset_id_from_info(item)
            if nested is not None:
                return nested
    return None


def _match_benchmark_type(records: List[Dict[str, str]], eval_type: int) -> Optional[Dict[str, str]]:
    target_type = EVAL_TYPE_MAP[eval_type].replace("_", " ")
    for record in records:
        record_type = str(record.get("type", "")).replace("_", " ").strip().lower()
        if record_type == target_type:
            return record
    return None


def _extract_dataset_id_from_benchmark(deployment_id: int, benchmark_id: int) -> Optional[int]:
    detail = benchmark_info(deployment_id, benchmark_id)
    return _safe_int(detail.get("info", {}).get("dataset_id"))


def _make_eval_config(eval_type: int, dataset_id: int) -> str:
    src = EVAL_CONFIG_MAP[eval_type]
    with open(src, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["dataset_id"] = dataset_id

    os.makedirs(ADELA_EVAL_CONFIG_DIR, mode=0o755, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
        dir=ADELA_EVAL_CONFIG_DIR,
        prefix="adela_eval_",
    )
    try:
        with temp:
            json.dump(config, temp, ensure_ascii=False, indent=2)
            temp.flush()
            try:
                os.fsync(temp.fileno())
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(temp.name)
        except OSError:
            pass
        raise
    return temp.name


def _emit(payload: Dict[str, Any]) -> None:
    print(f"data: {json.dumps(payload, ensure_ascii=False)}")
    print("")


def _emit_api(command: str, **kwargs: Any) -> None:
    """每执行一次底层 adela 命令后立即回流，便于前端逐步打勾与排障。"""
    payload: Dict[str, Any] = {"event": "adela_api_result", "command": command}
    for key, val in kwargs.items():
        if val is not None:
            payload[key] = val
    _emit(payload)


def _emit_done() -> None:
    print("data: [DONE]")


def _poll_deployment_terminal(deployment_id: int) -> Optional[Dict[str, Any]]:
    for i in range(MAX_POLL_COUNT):
        info = deployment_info(deployment_id)
        _emit_api(
            "deployment_info",
            deployment_id=deployment_id,
            poll_iteration=i + 1,
            status=str(info.get("status", "")),
        )
        status = str(info.get("status", "")).upper()
        if status in ("SUCCESS", "FAILURE"):
            return info
        if i < MAX_POLL_COUNT - 1:
            time.sleep(DEPLOYMENT_POLL_INTERVAL_SECONDS)
    return None


def _poll_benchmark_result(deployment_id: int, benchmark_id: int) -> Optional[Dict[str, Any]]:
    """
    发起评测后轮询 benchmark_info：每 30 秒查一次，最多 30 次。
    仅当 info.status 为 SUCCESS 或 FAILURE 时结束并返回详情；
    STARTED 等非终态继续轮询直至达到次数上限。
    """
    for i in range(MAX_POLL_COUNT):
        detail = benchmark_info(deployment_id, benchmark_id)
        status = str(detail.get("info", {}).get("status", "")).upper()
        _emit_api(
            "benchmark_info",
            deployment_id=deployment_id,
            benchmark_id=benchmark_id,
            poll_iteration=i + 1,
            status=status,
        )
        if status in ("SUCCESS", "FAILURE"):
            return detail
        if i < MAX_POLL_COUNT - 1:
            time.sleep(BENCHMARK_POLL_INTERVAL_SECONDS)
    return None


def _normalize_platform_key(value: str) -> str:
    """与 CLI 表格里 platform 列对齐：去首尾空白、统一小写、去掉中间空白便于比较。"""
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _platform_matches(record_platform: str, wanted: str) -> bool:
    a = _normalize_platform_key(record_platform)
    b = _normalize_platform_key(wanted)
    if not b:
        return False
    return a == b


def _find_target_platform_deployment(records: List[Dict[str, str]], platform: str) -> Optional[int]:
    """取首个 SUCCESS 且 platform 与目标串（宽松匹配）一致的部署 did。"""
    for record in records:
        if str(record.get("status", "")).upper() != "SUCCESS":
            continue
        if not _platform_matches(str(record.get("platform", "")), platform):
            continue
        did = _safe_int(record.get("did"))
        if did is not None:
            return did
    return None


def run_pipeline(rawmodel_id: int, platform: str, eval_type: int) -> Dict[str, Any]:
    if eval_type not in EVAL_TYPE_MAP:
        raise ValueError("eval_type 仅支持 0(精度) 或 1(性能)")

    platform = str(platform or "").strip()
    is_quant = _is_quant_platform(platform)
    all_deployments = deployment_list(rawmodel_id)
    target_deployment_id = _find_target_platform_deployment(all_deployments, platform)
    quant_dataset_step_needed = target_deployment_id is None and is_quant
    _emit(
        {
            "event": "deployment_list_result",
            "command": "deployment_list",
            "rawmodel_id": rawmodel_id,
            "platform": platform,
            "is_quant_platform": is_quant,
            "deployment_count": len(all_deployments),
            "target_deployment_id": target_deployment_id,
            "quant_dataset_step_needed": quant_dataset_step_needed,
        }
    )

    quant_dataset_id: Optional[int] = None

    # Step1: 目标平台已有部署时，先直接查历史评测结果
    if target_deployment_id is not None:
        target_benchmarks = benchmark_list(target_deployment_id)
        target_benchmarks_success = [
            b for b in target_benchmarks if str(b.get("status", "")).upper() == "SUCCESS"
        ]
        matched = _match_benchmark_type(target_benchmarks_success, eval_type)
        bid_probe: Optional[int] = _safe_int(matched.get("bid")) if matched else None
        _emit(
            {
                "event": "benchmark_probe_result",
                "command": "benchmark_list",
                "deployment_id": target_deployment_id,
                "record_count": len(target_benchmarks),
                "matched": matched is not None and bid_probe is not None,
                "benchmark_id": bid_probe,
            }
        )
        if matched:
            bid = _safe_int(matched.get("bid"))
            if bid is None:
                raise RuntimeError("命中历史评测但无法解析 benchmark_id")
            detail = benchmark_info(target_deployment_id, bid)
            _emit_api("benchmark_info", deployment_id=target_deployment_id, benchmark_id=bid)
            info = detail.get("info") if isinstance(detail.get("info"), dict) else {}
            dataset_id_hit = _safe_int(info.get("dataset_id"))
            _emit(
                {
                    "event": "adela_existing_result",
                    "platform": platform,
                    "deployment_id": target_deployment_id,
                    "benchmark_id": bid,
                    "dataset_id": dataset_id_hit,
                    "result": detail,
                }
            )
            return {
                "message": " ".join(
                    [f"部署ID {target_deployment_id}", f"benchmark_id {bid}"]
                    + ([f"dataset_id {dataset_id_hit}"] if dataset_id_hit is not None else [])
                ),
                "from_history": True,
                "deployment_id": target_deployment_id,
                "benchmark_id": bid,
                "dataset_id": dataset_id_hit,
                "result": detail,
            }
    else:
        _emit(
            {
                "event": "benchmark_probe_result",
                "deployment_id": None,
                "record_count": 0,
                "matched": False,
                "benchmark_id": None,
            }
        )
        # 目标平台不存在时，若是量化平台则检查历史量化数据集
        if is_quant:
            for d in all_deployments:
                p = str(d.get("platform", ""))
                t = str(d.get("type", ""))
                if not _is_quant_platform(p) and not _is_quant_platform(t):
                    continue
                did = _safe_int(d.get("did"))
                if did is None:
                    continue
                # 列表里已有与指定 platform 一致的部署时，不对该 did 做 deployment_info：
                # 该部署即「目标平台模型」，不应再当「借量化数据集」的数据源去扫详情。
                if _platform_matches(p, platform):
                    continue
                info = deployment_info(did)
                _emit_api("deployment_info", deployment_id=did, phase="quant_dataset_lookup")
                qid = _find_quant_dataset_id_from_info(info)
                if qid is not None:
                    quant_dataset_id = qid
                    break
            if quant_dataset_id is None:
                _emit(
                    {
                        "event": "quant_dataset_missing",
                        "message": "缺少量化数据集",
                    }
                )
                return {"message": "缺少量化数据集"}
            _emit(
                {
                    "event": "quant_dataset_result",
                    "dataset_id": quant_dataset_id,
                }
            )

    # Step3: 若目标部署无可用评测，遍历历史部署寻找可复用评测数据集；优先查目标平台部署，减少无关 CLI。
    dataset_id: Optional[int] = None
    eval_scan_order: List[Dict[str, str]] = list(all_deployments)
    if target_deployment_id is not None:
        first: List[Dict[str, str]] = []
        rest: List[Dict[str, str]] = []
        for dep in eval_scan_order:
            if _safe_int(dep.get("did")) == target_deployment_id:
                first.append(dep)
            else:
                rest.append(dep)
        eval_scan_order = first + rest

    for dep in eval_scan_order:
        if str(dep.get("status", "")).upper() != "SUCCESS":
            continue
        did = _safe_int(dep.get("did"))
        if did is None:
            continue
        records = benchmark_list(did)
        _emit_api("benchmark_list", deployment_id=did, record_count=len(records), phase="eval_dataset_lookup")
        success_records = [r for r in records if str(r.get("status", "")).upper() == "SUCCESS"]
        matched = _match_benchmark_type(success_records, eval_type)
        if matched:
            bid = _safe_int(matched.get("bid"))
            if bid is None:
                continue
            dataset_id = _extract_dataset_id_from_benchmark(did, bid)
            if dataset_id is not None:
                _emit_api("benchmark_info", deployment_id=did, benchmark_id=bid, phase="eval_dataset_lookup")
                break

    if dataset_id is None:
        _emit(
            {
                "event": "eval_dataset_missing",
                "message": "缺少评测数据集",
            }
        )
        return {"message": "缺少评测数据集"}

    _emit(
        {
            "event": "eval_dataset_result",
            "dataset_id": dataset_id,
        }
    )

    # Step4: 已有目标平台 SUCCESS 部署、但 Step1 未命中同类型评测，且已拿到可复用 dataset_id 时，
    # 不再发起新部署，直接在该 deployment 上发起评测并轮询（与其它路径相同的 30 秒 × 30 次）。
    reuse_existing_platform_deployment = target_deployment_id is not None

    if reuse_existing_platform_deployment:
        new_deployment_id = target_deployment_id
    else:
        add_rsp = deployment_add(
            rawmodel_id=rawmodel_id,
            platform=platform,
            quant_dataset_id=quant_dataset_id,
        )
        new_deployment_id = _extract_deployment_id(add_rsp)
        if new_deployment_id is None:
            raise RuntimeError("发起部署成功但未解析到 deployment_id")
        _emit_api("deployment_add", deployment_id=new_deployment_id, rawmodel_id=rawmodel_id, platform=platform)
        _emit(
            {
                "event": "submit_model_deployment",
                "deployment_id": new_deployment_id,
                "result": add_rsp,
            }
        )

        deploy_detail = _poll_deployment_terminal(new_deployment_id)
        if deploy_detail is None:
            return {
                "message": "部署超时",
                "deployment_id": new_deployment_id,
            }

        deploy_status = str(deploy_detail.get("status", "")).upper()
        _emit(
            {
                "event": "model_deployment_result",
                "deployment_id": new_deployment_id,
                "status": deploy_status,
                "result": deploy_detail,
            }
        )
        if deploy_status == "FAILURE":
            return {
                "message": "部署失败",
                "deployment_id": new_deployment_id,
                "detail": deploy_detail,
            }

    config_path = _make_eval_config(eval_type, dataset_id)
    benchmark_rsp = benchmark_add(new_deployment_id, config_path)
    benchmark_id = _extract_benchmark_id(benchmark_rsp)
    if benchmark_id is None:
        raise RuntimeError("发起评测成功但未解析到 benchmark_id")
    _emit_api("benchmark_add", deployment_id=new_deployment_id, benchmark_id=benchmark_id)
    _emit(
        {
            "event": "submit_model_evaluation",
            "deployment_id": new_deployment_id,
            "benchmark_id": benchmark_id,
            "result": benchmark_rsp,
        }
    )

    benchmark_result = _poll_benchmark_result(new_deployment_id, benchmark_id)
    if benchmark_result is None:
        return {
            "message": "评测超时",
            "detail": "约 15 分钟内（每 30 秒查询 1 次，最多 30 次）status 未变为 SUCCESS 或 FAILURE",
            "reuse_existing_platform_deployment": reuse_existing_platform_deployment,
            "deployment_id": new_deployment_id,
            "benchmark_id": benchmark_id,
        }

    final_status = str(benchmark_result.get("info", {}).get("status", "")).upper()
    _emit(
        {
            "event": "model_evluation_result",
            "deployment_id": new_deployment_id,
            "benchmark_id": benchmark_id,
            "status": final_status,
            "result": benchmark_result,
        }
    )
    if final_status == "FAILURE":
        return {
            "message": "评测失败",
            "reuse_existing_platform_deployment": reuse_existing_platform_deployment,
            "deployment_id": new_deployment_id,
            "benchmark_id": benchmark_id,
            "result": benchmark_result,
        }

    return {
        "message": "精度/性能结果",
        "from_history": False,
        "reuse_existing_platform_deployment": reuse_existing_platform_deployment,
        "is_quant_platform": is_quant,
        "quantify_dataset_id": quant_dataset_id,
        "deployment_id": new_deployment_id,
        "benchmark_id": benchmark_id,
        "dataset_id": dataset_id,
        "result": benchmark_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模型部署与评测自动流程")
    parser.add_argument("--rawmodel_id", type=int, required=True, help="原模型 ID")
    parser.add_argument("--platform", type=str, required=True, help="目标平台")
    parser.add_argument(
        "--eval_type",
        type=int,
        choices=[0, 1],
        required=True,
        help="0=精度评测, 1=性能评测",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        "[adela-cli] CMD:",
        " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        flush=True,
    )
    try:
        result = run_pipeline(
            rawmodel_id=args.rawmodel_id,
            platform=args.platform,
            eval_type=args.eval_type,
        )
        _emit(
            {
                "event": "adela_final_result",
                "message": str(result.get("message") or ""),
                "result": result,
            }
        )
    except Exception as e:  # noqa: BLE001
        error_message = str(e).strip()
        user_message = (
            "Adela CLI 执行失败，暂未获取到评测结果。"
            "请稍后重试，或检查 Adela 服务/部署记录输出是否正常。"
        )
        _emit(
            {
                "event": "adela_pipeline_error",
                "message": user_message,
                "error_detail": error_message,
            }
        )
    finally:
        _emit_done()
