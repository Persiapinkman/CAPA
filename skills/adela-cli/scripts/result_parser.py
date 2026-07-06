import json
import re
import ast
from typing import Any, Dict, List

# 部署列表续行里，最后一列常与 platform 尾（如 T4）折到同一行
_DEPLOY_STATUS_TOKENS = frozenset(
    {
        "SUCCESS",
        "FAILURE",
        "RUNNING",
        "STARTED",
        "PENDING",
        "STOPPED",
        "UNKNOWN",
        "CANCELLED",
        "CANCELED",
    }
)


def parse_deployment_list_output(output: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    primary_row_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+")

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if "部署列表(" in line or "DID" in line or set(line.strip()) == {"━"}:
            continue

        stripped = line.strip()
        segments = [seg for seg in re.split(r"\s{3,}", stripped) if seg]
        if not segments:
            continue

        if primary_row_pattern.match(line):
            if len(segments) >= 7:
                records.append(
                    {
                        "did": segments[0],
                        "rid": segments[1],
                        "type": segments[2],
                        "model_name": segments[3],
                        "version": segments[4],
                        "platform": segments[5],
                        "status": segments[6],
                    }
                )
        elif records:
            last = records[-1]
            version_is_complete = bool(re.fullmatch(r"\d+\.\d+\.\d+-\d{8}", last["version"]))
            # 折行：… int8-  |  T4    SUCCESS  → platform 补 T4，不把 SUCCESS 拼进 platform
            if (
                len(segments) >= 2
                and version_is_complete
                and segments[-1].strip().upper() in _DEPLOY_STATUS_TOKENS
            ):
                for part in segments[:-1]:
                    last["platform"] += part
                continue
            if len(segments) >= 1:
                # 仅一个片段且 version 已齐：多为 platform 尾（如 T4）折行，避免误拼到 model_name
                if (
                    len(segments) == 1
                    and version_is_complete
                    and last["platform"].rstrip().endswith("-")
                    and re.fullmatch(r"[A-Za-z0-9+.-]{1,24}", segments[0])
                ):
                    last["platform"] += segments[0]
                    continue
                last["model_name"] += segments[0]
            if len(segments) >= 2:
                if not version_is_complete:
                    last["version"] += segments[1]
                else:
                    last["platform"] += segments[1]
            if len(segments) >= 3:
                last["platform"] += segments[2]

    return records


def print_records_json(records: List[Dict[str, str]]) -> None:
    if not records:
        print("未解析到部署记录。")
        return

    print(json.dumps(records, ensure_ascii=False, indent=2))


def _replace_datetime_repr(raw: str) -> str:
    pattern = re.compile(
        r"datetime\.datetime\("
        r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*tzinfo=tzutc\(\)\)"
    )

    def repl(match: re.Match) -> str:
        year, month, day, hour, minute, second = [int(v) for v in match.groups()]
        return f"'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z'"

    return pattern.sub(repl, raw)


def _normalize_wrapped_single_quote_strings(raw: str) -> str:
    """
    adela 的 info 输出里，超长字符串有时会被硬换行。
    这种换行会让 ast.literal_eval 报 "unterminated string literal"。
    这里在“字符串内部”遇到真实换行时直接去掉，恢复成可解析形式。
    """
    normalized: List[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for ch in raw:
        if (in_single_quote or in_double_quote) and ch == "\n":
            # 去掉字符串内部的硬换行（终端折行产物）
            continue

        normalized.append(ch)

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

    return "".join(normalized)


def _literal_eval_mixed_output(output: str) -> Any:
    raw = str(output or "").strip()
    if not raw:
        raise ValueError("adela 输出为空")

    candidates = [raw]
    for marker in ("Error!!!", "Traceback (most recent call last):"):
        idx = raw.find(marker)
        if idx > 0:
            candidates.append(raw[:idx].strip())

    for candidate in candidates:
        cleaned = _replace_datetime_repr(candidate)
        cleaned = _normalize_wrapped_single_quote_strings(cleaned)
        try:
            return ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            continue

    last_error = None
    for start, end in (("{", "}"), ("[", "]")):
        begin = raw.find(start)
        if begin < 0:
            continue
        depth = 0
        in_single = False
        in_double = False
        escaped = False
        for idx, ch in enumerate(raw[begin:], start=begin):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if in_double:
                if ch == '"':
                    in_double = False
                continue
            if ch == "'":
                in_single = True
                continue
            if ch == '"':
                in_double = True
                continue
            if ch == start:
                depth += 1
                continue
            if ch == end:
                depth -= 1
                if depth == 0:
                    snippet = raw[begin : idx + 1]
                    cleaned = _replace_datetime_repr(snippet)
                    cleaned = _normalize_wrapped_single_quote_strings(cleaned)
                    try:
                        return ast.literal_eval(cleaned)
                    except (SyntaxError, ValueError) as exc:
                        last_error = exc
                        break

    if "Traceback (most recent call last):" in raw or "Error!!!" in raw:
        raise ValueError("Adela CLI 返回了异常输出，无法解析有效结果")
    if last_error is not None:
        raise ValueError(f"无法解析 Adela CLI 输出: {last_error}") from last_error
    raise ValueError("无法解析 Adela CLI 输出")


def parse_deployment_info_output(output: str) -> Dict[str, Any]:
    data = _literal_eval_mixed_output(output)

    for key in ("meta_json", "param_json", "config_json"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return data


def parse_benchmark_list_output(output: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if "评测列表(" in line or "BID" in line or set(line.strip()) == {"━"}:
            continue

        segments = [seg for seg in re.split(r"\s{3,}", line.strip()) if seg]
        if len(segments) >= 4 and segments[0].isdigit():
            records.append(
                {
                    "bid": segments[0],
                    "type": segments[1],
                    "dataset": segments[2],
                    "status": segments[3],
                }
            )

    return records


def parse_benchmark_info_output(output: str) -> Dict[str, Any]:
    info_marker = "=============Info============="
    result_marker = "=============Result============="

    info_idx = output.find(info_marker)
    result_idx = output.find(result_marker)
    if info_idx == -1 or result_idx == -1:
        raise ValueError("benchmark info 输出中未找到 Info/Result 段落")

    info_block = output[info_idx + len(info_marker):result_idx].strip()
    result_block = output[result_idx + len(result_marker):].strip()

    info = _literal_eval_mixed_output(info_block)
    try:
        result = ast.literal_eval(result_block)
    except (SyntaxError, ValueError):
        # 某些 benchmark result 为表格文本，不是 Python 字面量
        result = result_block

    extra = info.get("extra")
    if isinstance(extra, str) and extra.strip().startswith("{") and extra.strip().endswith("}"):
        try:
            info["extra"] = ast.literal_eval(
                _normalize_wrapped_single_quote_strings(extra)
            )
        except (ValueError, SyntaxError):
            pass

    return {
        "info": info,
        "result": result,
    }


def parse_benchmark_add_output(output: str) -> Dict[str, Any]:
    data = _literal_eval_mixed_output(output)

    extra = data.get("extra")
    if isinstance(extra, str) and extra.strip().startswith("{") and extra.strip().endswith("}"):
        try:
            data["extra"] = ast.literal_eval(extra)
        except (ValueError, SyntaxError):
            pass

    return data


def print_json(data: Any) -> None:
    if not data:
        print("未解析到有效结果。")
        return
    print(json.dumps(data, ensure_ascii=False, indent=2))
